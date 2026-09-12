"""Learned Q4 waypoint/action decisions with explicit survey macro semantics."""
import time
from dataclasses import asdict
import torch
from .bridge import ProbePolicy, Config
from .state import MenuPlanner
from .network import PolicyNetwork


class NeuralPolicy:
    def __init__(self, checkpoint=None, device='cpu', macro_budget=160,
                 allow_fallback=True, teacher=False, selection='policy', config=None,
                 sample=False, sampling_seed=42, temperature=1., menu='full', layout='ring22', features='v1'):
        self.planner = MenuPlanner(config, menu=menu, layout=layout, features=features)
        self.teacher = teacher
        self.network = None
        self.device = torch.device(device)
        self.macro_budget = macro_budget
        self.allow_fallback = allow_fallback
        self.selection = selection
        self.sample = sample
        self.temperature = temperature
        self.generator = torch.Generator(device=device).manual_seed(sampling_seed)
        self.last_frame = None
        self.last_decision = None
        self.macro_owner = None
        self.stats = dict(neural_decisions=0, teacher_decisions=0, macro_continuation_actions=0,
                          fallback_actions=0, forward_s=0., encode_menu_s=0.)
        if checkpoint:
            if selection == 'advantage':
                from .advantage import AdvantageNetwork
                if (menu, layout, features) != ('full', 'ring22', 'v2'):
                    raise ValueError('Advantage model requires full/ring22/v2')
                self.network = AdvantageNetwork.load(checkpoint, device)
                self.stats.update(network_overrides=0, baseline_decisions=0)
                return
            if selection == 'relative':
                from .relative import RelativeCostNetwork
                if (menu, layout, features) != ('full', 'ring22', 'v2'):
                    raise ValueError('First-round relative model requires full/ring22/v2')
                self.network = RelativeCostNetwork.load(checkpoint, device)
                self.stats.update(network_overrides=0, baseline_decisions=0)
                return
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            network_config = saved['network_config'] if 'network_config' in saved else saved['config']
            self.network = PolicyNetwork.from_config(network_config).to(self.device)
            self.network.load_state_dict(saved['model'] if 'model' in saved else saved['model_state_dict'])
            self.network.eval()

    def choose(self, b):
        self.last_frame = None
        if b.done():
            raise ValueError('Action requested after public completion')
        count = self.stats['neural_decisions'] + self.stats['teacher_decisions']
        if count >= self.macro_budget or b.deadline-time.monotonic() < 35:
            if not self.allow_fallback:
                raise RuntimeError('Pure neural evaluation exhausted its macro/time budget')
            self.stats['fallback_actions'] += 1
            action = self.planner.choose(b)
            self.last_decision = {'executor': 'fallback', 'reason': 'macro_or_time_budget',
                                  'reference_action': asdict(action), 'selected_action': asdict(action)}
            return action
        continuation = self.planner.pending_action(b)
        if continuation is not None:
            self.stats['macro_continuation_actions'] += 1
            self.last_decision = {'executor': 'macro_continuation', 'reason': 'selected_survey_sequential_channels',
                                  'macro_owner': self.macro_owner, 'selected_action': asdict(continuation)}
            return continuation
        start = time.monotonic()
        frame = self.planner.frame(b)
        self.stats['encode_menu_s'] += time.monotonic()-start
        self.last_frame = frame
        if self.teacher:
            selected = frame.target
            self.stats['teacher_decisions'] += 1
            executor = 'teacher'
        elif self.selection in ('relative', 'advantage'):
            start = time.monotonic()
            selected, relative_detail = self.network.choose_frame(frame)
            self.stats['forward_s'] += time.monotonic()-start
            self.stats['neural_decisions'] += 1
            executor = 'network_override' if selected != frame.target else 'baseline'
            self.stats['network_overrides' if selected != frame.target else 'baseline_decisions'] += 1
        else:
            start = time.monotonic()
            arrays = frame.tensors()
            data = [arrays[k][None].to(self.device) for k in
                    ('nodes', 'node_mask', 'candidates', 'candidate_mask', 'global_features')]
            with torch.inference_mode():
                out = self.network(*data)
                logits = out['scores'][0]/self.temperature
                if self.sample:
                    selected = int(torch.multinomial(logits.softmax(-1), 1, generator=self.generator))
                else:
                    selected = int(out['q'][0].argmin()) if self.selection == 'q' else int(out['scores'][0].argmax())
                log_probability = float(logits.log_softmax(-1)[selected])
            self.stats['forward_s'] += time.monotonic()-start
            self.stats['neural_decisions'] += 1
            executor = 'network'
        action = frame.actions[selected]
        self.planner.commit(action, frame.survey[selected])
        self.macro_owner = executor
        self.last_decision = {'executor': executor, 'reason': 'candidate_node_pointer',
                             'candidates': len(frame.actions), 'selected': selected,
                             'survey_macro': frame.survey[selected], 'reference': frame.target,
                             'matches_reference': selected == frame.target}
        self.last_decision['candidate_family'] = frame.families[selected]
        if self.selection in ('relative', 'advantage') and not self.teacher:
            self.last_decision.update(relative_detail)
            if self.selection == 'relative':
                self.last_decision['reason'] = ('conservative_relative_cost_override' if selected != frame.target
                                                else 'relative_cost_threshold_retain_baseline')
        elif not self.teacher:
            self.last_decision.update(log_probability=log_probability, sampled=self.sample,
                                      temperature=self.temperature)
        return action
