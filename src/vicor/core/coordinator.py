import asyncio
import copy
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .actions import ActionArbiter
from ..backends.model_client import MAIN_GAME_TOOLS, TEXT, NUMBER, tool

from .roles import build_gameworld_role_profile, resolve_semantic_action
from .session_context import DEFAULT_LONG_CONTEXT_TRIGGER_TOKENS, SessionContext
from .streams import Events, FrameStream
from .visual_reflex import TRIGGER_CONTRACT, VisualReflexEngine
from .archive import VisualArchive


INPUT_TOOLS = {'act', 'press', 'press_keys', 'press_sequence', 'click', 'click_grid', 'wait'}
INTERMISSION_TOOLS = {'set_reflex', 'drop_reflex', 'list_reflexes', 'restore_best_reflexes',
                      'inspect', 'read_pixels', 'visual_history', 'write_note',
                      'get_action_history'}


def annotated_view(raw):
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, 100):
        draw.line((x, 0, x, 10), fill=(220, 50, 50), width=1)
        draw.text((x + 2, 10), str(x), fill=(220, 50, 50), stroke_width=1, stroke_fill='white')
    for y in range(0, image.height, 100):
        draw.line((0, y, 10, y), fill=(220, 50, 50), width=1)
        draw.text((12, y + 2), str(y), fill=(220, 50, 50), stroke_width=1, stroke_fill='white')
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


class VisualReflexMain:
    def __init__(self, client, frames, actions, engine, events, profile, instruction,
                 tokenizer_file, effort='low', compaction_threshold=100000,
                 game_env=None, baseline_style='vista'):
        self.client, self.frames, self.actions, self.engine = client, frames, actions, engine
        self.events, self.profile = events, profile
        self.env = game_env
        self.pause_during_inference = bool(
            game_env is not None and game_env.config.pause_during_inference)
        if baseline_style not in {'react', 'reflexion', 'vista'}:
            raise ValueError('baseline_style must be react, reflexion or vista')
        if baseline_style != 'vista' and engine.mode != 'pure-vlm':
            raise ValueError('react/reflexion are pure-VLM, without a reflex library')
        self.baseline_style = baseline_style
        self.instruction, self.effort = instruction, effort
        self.archive = VisualArchive(events, engine.directory / 'notes')
        self.session_context = SessionContext(engine.directory / 'context' / 'main', tokenizer_file,
                                              events, trigger_tokens=compaction_threshold, context_limit=131072)
        self.reviews = []
        self.next_capture_after = None
        self.closed = False
        self.decision_lock = asyncio.Lock()
        self.tool_specs = self._tools()
        self.system = self._prompt()

    def _prompt(self):
        common = (
            'You are the single autonomous game agent. Achieve the original task through real visual interaction. '
            + ('The game clock advances only during executed inputs or explicit waits; all other time is paused. '
               if self.pause_during_inference else
               'The world CONTINUES RUNNING while you think; inference latency is part of the game. ')
            + 'Each decision automatically receives a NEW full screenshot; '
            'after a real input it is captured AFTER that input. There is no get_screenshot round trip. '
            'Use original image pixel coordinates. '
                'Only you start each ready game. Start by an actual permitted click/key from the visible screen. '
                'Native game outcomes alone determine official success or failure. '
                + ('The game resets without an extra between-round learning turn. '
                   if self.engine.mode == 'pure-vlm' and self.baseline_style == 'react'
                   else 'Before reset you receive the official current score, objective score/target, '
                        'and the last five pre-terminal frames in a separate review turn. ')
                +
                'Act on current images, not old positions. Avoid long narration; use tools. '
        )
        if self.engine.mode != 'pure-vlm':
            common += (
                'You write/update self-triggered reflex memory yourself; '
                'there is no apprentice or coding subagent. '
                + ('Use set_reflex(name, trigger_py, memory_40w, direct_action), '
                   'where a Python trigger immediately executes the selected native direct_action. '
                   'There is NO GPU0 fast-model decision in this Python-only baseline. '
                   if self.engine.mode == 'pure-python' else
                   'Use set_reflex(name, trigger_py, memory_40w). GPU0 runs the '
                   'SAME underlying model as you, with the same visual and game-understanding '
                   'capability, merely in fast no-thinking mode. It receives the GameWorld rules, '
                   'the complete scene, and ALL legal native actions plus NONE. '
                   'Trust it to make rapid decisions, including rejecting false alerts with NONE; '
                   'actively publish and improve reflexes rather than deferring useful ideas '
                   'because you fear it might choose a wrong direction. ')
                + 'You remain responsible for long-horizon strategy and can still act directly. '
                + TRIGGER_CONTRACT + ' '
                'RED BOX marks the object/region that Python actually matched, never a fixed hand-drawn ROI. '
                'MOTION GHOST is a faint 30%-opacity trace of pixels around 0.25s earlier: '
                'the shift toward solid pixels suggests motion direction/speed and vertical movement can reveal '
                'airborne/falling motion; account for moving cameras. '
                + ('' if self.engine.mode == 'pure-python' else
                   'GPU0 sees the complete 640x360 game image with those overlays and chooses from ALL native '
                   'executable GameWorld actions plus NONE; a NONE is silent with ZERO cooldown. '
                   'A real reflex action briefly takes priority without suppressing later sampling '
                   'or cancelling subsequent main commands; only one GPU0 inference runs at once. ')
                + 'Use inspect/read_pixels to study images before writing check; registration returns a '
                'dry-run full-scene preview, including match boxes if currently triggered. '
                'Continue to control the game directly for planning and normal actions. '
                'After each native round, there is a dedicated intermission BEFORE the next game starts: '
                'you receive the official score/target and five pre-terminal frames; '
                'reflect on repeated failures and publish, debug, update, or drop reflex code there. '
                'Do not settle for writing strategy notes without taking useful learning actions. '
            )
        if self.engine.mode == 'pure-python':
            common += ('This is the direct-Python comparison: no fast visual model is used. '
                       'The matched trigger immediately executes the declared native direct_action. '
                       'All other observation and control tools are unchanged. ')
        if self.engine.mode == 'pure-vlm':
            common += 'This comparison arm has no reflex tools; pursue the goal with direct controls and visual inspection. '
            if self.baseline_style == 'react':
                common += ('ReAct-style: interleave your own visual reasoning and direct game actions. '
                           'There is no separate between-round reflection turn and no persistent notes '
                           'or historical visual retrieval tools. ')
            elif self.baseline_style == 'reflexion':
                common += ('Reflexion-style: at the dedicated end-of-round review, explicitly examine '
                           'official outcome feedback and update GUIDE.md with concrete lessons. '
                           'Apply your verbal memory in later rounds; no executable reflex code or '
                           'historical visual retrieval is available. ')
            else:
                common += ('VISTA-style: use inspect/read_pixels/visual_history to actively examine '
                           'losslessly preserved visual evidence. Keep reusable understanding in '
                           'GUIDE.md and current plans in WORKING.md; review genuine outcomes. ')
        elif self.baseline_style == 'vista':
            common += ('Use inspect/read_pixels/visual_history for lossless visual history, '
                       'keep GUIDE.md as reusable memory and WORKING.md for current plans. ')
        if self.engine.bank_enabled and self.engine.mode != 'pure-vlm':
            common += ('The harness preserves immutable card versions and the best completed-round library. '
                       'Two lower-scoring rounds with a changed library restore that champion; you receive a receipt. ')
        return common + '\nUser task instruction: ' + self.instruction + '\nOriginal GameWorld rules and controls:\n' + self.profile.rules_text

    def _tools(self):
        tools = [x for x in MAIN_GAME_TOOLS if x['function']['name'] in
                 {'press', 'press_keys', 'press_sequence', 'click', 'click_grid', 'get_action_history'}]
        tools += [
            tool('act', 'Perform one official semantic action with its declared parameters.',
                 {'action_id': {'type': 'string', 'enum': list(self.profile.semantic)},
                  'parameters': {'type': 'object'}}, ('action_id',)),
            tool('wait', 'Briefly leave the controls unchanged; a fresh image arrives afterwards.',
                 {'duration_ms': NUMBER}),
            tool('inspect', 'Inspect the full ORIGINAL screenshot with pixel coordinate ticks, or zoom into a region. '
                 'The current frame ID is given in each observation.',
                 {'frame_id': NUMBER, 'question': TEXT,
                  'region': {'type': 'object', 'properties': {k: NUMBER for k in ('x', 'y', 'width', 'height')},
                             'required': ['x', 'y', 'width', 'height']}}, ('frame_id',)),
            tool('read_pixels', 'Read exact RGB samples and image size from original pixels.',
                 {'frame_id': NUMBER, 'x': NUMBER, 'y': NUMBER, 'width': NUMBER, 'height': NUMBER},
                 ('frame_id', 'x', 'y', 'width', 'height')),
            tool('visual_history', 'Read saved frame IDs and actual inputs/terminal outcomes.', {'limit': NUMBER}),
            tool('write_note', 'Save concise revisable experience or the current plan.',
                 {'name': {'type': 'string', 'enum': ['GUIDE.md', 'WORKING.md']}, 'content': TEXT}, ('name', 'content')),
        ]
        if self.engine.mode != 'pure-vlm':
            tools += [tool('set_reflex', 'Create or revise a visual reflex. ' + TRIGGER_CONTRACT
                            + ' The fast instance reads the marked full scene, confirms the learned cue '
                            'and chooses one native response or NONE.',
                           {'name': TEXT,
                            'trigger_py': TEXT, 'memory_40w': TEXT,
                            **({'direct_action': TEXT} if self.engine.mode == 'pure-python' else {})},
                           ('name', 'trigger_py', 'memory_40w')),
                      tool('drop_reflex', 'Remove a harmful or obsolete card immediately.', {'name': TEXT}, ('name',)),
                      tool('list_reflexes', 'Inspect installed cards and actual trigger/action/NONE counters.')]
            if self.engine.bank_enabled:
                tools.append(tool('restore_best_reflexes', 'Restore the library from the highest native score completed round.'))
        if getattr(self, 'baseline_style', 'vista') in {'react', 'reflexion'}:
            # A disciplined published-method adaptation, not a fake VISTA run:
            # ReAct: current image + reasoning/actions; Reflexion adds notes.
            allowed = INPUT_TOOLS | {'get_action_history'}
            if self.baseline_style == 'reflexion':
                allowed.add('write_note')
            tools = [spec for spec in tools
                     if spec['function']['name'] in allowed]
        return tools

    def add_review(self, record, frames):
        self.reviews.append((record, frames))

    async def _ask_main(self, messages, options, frame):
        # ActionArbiter manages the game clock without interrupting active short inputs.
        return await self.client.ask(
            messages, options, role='main', max_tokens=4096, effort=self.effort)

    async def _tool(self, name, args, frame, epoch):
        if name == 'set_reflex':
            result = self.engine.set_reflex(**args)
            # In an intermission, preview against the last *playing* frame,
            # not the GAME OVER overlay. During play this is the same frame.
            image, preview = await asyncio.to_thread(self.engine.preview, args['name'], frame)
            self.archive.pending_images.append((
                {'rule': args['name'], 'purpose': 'GPU0 full-scene 640x360 preview: '
                 'red boxes are the actual Python check() matches; ghost shows ~0.25s past motion',
                 'hit_boxes': preview['match_boxes'], 'dry_run_error': preview['dry_run_error'],
                 'size': [640, 360]},
                image))
            return {**result, **preview}
        if name == 'drop_reflex':
            answer = self.engine.drop_reflex(args['name'])
            await self.actions.disable_source('rule:' + args['name'])
            return answer
        if name == 'list_reflexes':
            return self.engine.list()
        if name == 'restore_best_reflexes':
            return self.engine.restore_best()
        if name == 'write_note':
            return self.archive.write_note(args['name'], args['content'])
        if name == 'visual_history':
            return self.archive.history(args.get('limit', 12))
        if name == 'get_action_history':
            return list(self.actions.history)[-12:]
        if name == 'inspect':
            number = int(args['frame_id'])
            if args.get('region') is not None:
                return self.archive.inspect(number, args['region'], args.get('question', 'Inspect the selected region'))
            record = self.archive.frames()[number]
            raw = Path(record['saved_path']).read_bytes()
            description = {'frame_id': number, 'coordinate_system': 'original image pixels',
                           'ticks_every_px': 100, 'original_size': list(Image.open(io.BytesIO(raw)).size)}
            self.archive.pending_images.append((description, annotated_view(raw)))
            return description
        if name == 'read_pixels':
            record = self.archive.frames()[int(args['frame_id'])]
            image = Image.open(record['saved_path']).convert('RGB')
            x, y, w, h = (int(args[k]) for k in ('x', 'y', 'width', 'height'))
            if not (0 <= x < x+w <= image.width and 0 <= y < y+h <= image.height):
                raise ValueError('Pixel region outside original image')
            samples = [{'x': px, 'y': py, 'rgb': list(image.getpixel((px, py)))}
                       for py in range(y, y+h, max(1, h//8))
                       for px in range(x, x+w, max(1, w//8))][:81]
            return {'original_size': list(image.size), 'samples': samples}
        if name not in INPUT_TOOLS:
            raise ValueError(f'Unknown tool {name!r}')
        if name == 'act':
            action_id = args['action_id']
            action = resolve_semantic_action(self.profile.semantic[action_id], self.profile.specs[action_id],
                                             args.get('parameters', {}))
        elif name in {'press', 'press_keys'}:
            key = 'key' if name == 'press' else 'keys'
            action = {'action': 'press_key' if name == 'press' else 'press_keys',
                      key: args[key], 'duration': args.get('duration_ms', 150)/1000}
        elif name == 'press_sequence':
            answer = await self.actions.run_sequence(args['steps'], source='main', frame=frame,
                                                     expected_epoch=epoch)
            self.next_capture_after = answer.get('last_finished_at')
            return answer
        elif name in {'click', 'click_grid'}:
            width, height = Image.open(io.BytesIO(frame.image)).size
            x, y = args['x'], args['y']
            if name == 'click_grid':
                if not 0 <= x <= 1000 or not 0 <= y <= 1000:
                    raise ValueError('Grid coordinates must be 0..1000')
                x, y = width*x/1000, height*y/1000
            action = {'action': 'click', 'x': x, 'y': y}
        else:
            action = {'action': 'wait', 'duration': args.get('duration_ms', 100)/1000}
        result = await self.actions.submit(action, source='main', frame=frame, expected_epoch=epoch)
        if result.get('finished_at') is not None:
            self.next_capture_after = result['finished_at']
        if result.get('status') == 'executing':
            completed = await self.actions.wait_for_completion(result['action_id'])
            self.next_capture_after = completed['monotonic']
            return {**result, 'status': 'completed' if completed['reason'] == 'duration_completed' else 'interrupted',
                    'reason': completed['reason'], 'finished_at': completed['monotonic']}
        return result

    def _observation(self, frame):
        saved = self.frames.save(frame, reason='main_observation')
        recent = [{k: row.get(k) for k in ('action_id', 'source', 'action', 'frame_id')}
                  for row in self.actions.history if row['event'] == 'action_started'
                  and row['episode_id'] == frame.episode_id][-3:]
        notices = self.engine.drain_notifications()
        images = []
        reviews = []
        while self.reviews:
            review, pictures = self.reviews.pop(0)
            reviews.append(review)
            images.extend(pictures)
        for index, (description, image) in enumerate(self.archive.pending_images):
            folder = self.events.directory / 'inspections'
            folder.mkdir(exist_ok=True)
            path = folder / f'{frame.frame_id:07d}_{index}.png'
            path.write_bytes(image)
            images.append((str(path), 'Requested inspection: ' + json.dumps(description)))
        self.archive.pending_images.clear()
        for notice in notices[-3:]:
            if notice.get('frame_path') and notice.get('episode_id') == frame.episode_id:
                images.append((notice['frame_path'], notice.get('message', notice['event'])))
        images.append((str(saved), f'CURRENT frame={frame.frame_id}, original {Image.open(io.BytesIO(frame.image)).size}, phase={frame.phase}'))
        state = {'episode_id': frame.episode_id, 'frame_id': frame.frame_id, 'phase': frame.phase,
                 'captured_at_monotonic': frame.timestamp, 'recent_actual_inputs': recent,
                 'notes': self.archive.notes(),
                 'installed_reflexes': [{'name': name, 'version': row['version'],
                                        'trigger': 'check(frame,prev_frame,state)'}
                                       for name, row in self.engine.cards.items()],
                 'reflex_receipts': notices[-3:], 'native_completed_round_reviews': reviews}
        self.events.add('main_context', frame_id=frame.frame_id, episode_id=frame.episode_id,
                        delivered_event_ids=[row['id'] for row in notices],
                        automatic_current_image=True, post_action_capture_after=self.next_capture_after)
        return json.dumps(state, ensure_ascii=False, default=str), images

    def _review_tools(self):
        """Allow active learning, not physical movement, before the next reset."""
        specs = [spec for spec in self.tool_specs
                 if spec['function']['name'] in INTERMISSION_TOOLS]
        specs.append(tool('finish_review',
                          'Finish intermission learning and let the next native game begin.',
                          {}))
        return specs

    @staticmethod
    def _review_score(record):
        metrics = record.get('metrics', {})
        objectives = metrics.get('objective_target_results', [])
        return {
            'current_score': metrics.get('score'),
            'official_objectives': objectives,
            'native_status': record.get('status'),
            'native_stop_reason': record.get('stop_reason'),
            'official_metrics': {
                key: value for key, value in metrics.items()
                if key in {'score', 'target_reached', 'progress_current',
                           'progress', 'objective_targets_met'}
            },
        }

    async def review_episode(self, record, evidence, review, observation):
        """A real model/tool learning turn *before* resetting the game.

        The usual playing loop and this review share a decision lock. The
        controller can inspect terminal evidence, debug/rewrite cards, and
        explicitly finish; no gameplay controls are exposed in this phase.
        """
        async with self.decision_lock:
            self.add_review(review, evidence)
            options = self._review_tools()
            turns = 0
            changes = 0
            for attempt in range(4):
                note, pictures = self._observation(observation)
                review_instruction = (
                    'Reflect on the official outcome and update verbal episodic lessons in GUIDE.md. '
                    'There are no executable reflexes in this method.'
                    if self.engine.mode == 'pure-vlm' else
                    'Reflect on failed/successful reactions and ACT by publishing, '
                    'modifying or debugging useful reflexes via set_reflex or related tools. '
                    'GPU0 has the same underlying model as you; trust it to choose a native '
                    'action or NONE. Examine actual hit boxes, not a fixed search strip.')
                note += '\n[DEDICATED BETWEEN-ROUND LEARNING REVIEW]\n' + json.dumps({
                    'round_finished': True,
                    'episode_id': record['episode_id'],
                    'score_and_official_result': self._review_score(record),
                    'current_reflexes': [
                        {'name': name, 'version': card['version']}
                        for name, card in self.engine.cards.items()
                    ],
                    'reflex_statistics': review.get('reflex_statistics', {}),
                    'review_turn': attempt+1,
                    'next_round_waits_for_this_review': True,
                    'instruction': review_instruction +
                                   ' Use finish_review when learning is complete.',
                }, ensure_ascii=False)
                messages = self.session_context.model_messages(self.system, note, pictures)
                estimate = self.session_context.estimate(messages, options, max_output=4096)
                if estimate >= self.session_context.context_limit - 6000:
                    self.session_context.schedule_if_needed(
                        self.client, system=self.system, tools=options,
                        output_tokens=4096, role='main')
                    if self.session_context.task:
                        await self.session_context.task
                    # Rebuild the message after compaction, without dropping
                    # the official result or the previous five evidence frames.
                    messages = self.session_context.model_messages(self.system, note, pictures)
                response = await self._ask_main(messages, options, observation)
                turns += 1
                calls = response.get('tool_calls') or []
                replies = []
                finished = False
                for call in calls:
                    fn = call['function']
                    name = fn['name']
                    args = {}
                    try:
                        args = (json.loads(fn['arguments'])
                                if isinstance(fn['arguments'], str) else fn['arguments'])
                        if name == 'finish_review':
                            answer = {'status': 'review_complete'}
                            finished = True
                        elif name not in INTERMISSION_TOOLS:
                            answer = {'status': 'not_executed',
                                      'reason': 'no_physical_controls_during_review'}
                        else:
                            answer = await self._tool(name, args, observation, self.actions.epoch)
                            if name in {'set_reflex', 'drop_reflex', 'restore_best_reflexes'} and (
                                    answer.get('status') in {'registered', 'dropped', 'restored'}):
                                changes += 1
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        answer = {'status': 'error',
                                  'error': f'{type(exc).__name__}: {exc}'}
                    self.events.add('main_tool', tool=name, arguments=args,
                                    result=answer, frame_id=observation.frame_id,
                                    episode_id=record['episode_id'], review_only=True)
                    replies.append({'role': 'tool', 'tool_call_id': call['id'],
                                    'content': json.dumps(answer, ensure_ascii=False, default=str)})
                self.session_context.append(note, pictures, response, replies)
                self.session_context.schedule_if_needed(
                    self.client, system=self.system, tools=options,
                    output_tokens=4096, role='main')
                if finished or not calls:
                    break
            self.events.add('reflex_intermission_review_completed',
                            episode_id=record['episode_id'], turns=turns,
                            card_changes=changes,
                            official_scores=self._review_score(record))
            return {'turns': turns, 'card_changes': changes}

    async def run(self):
        last = 0
        while not self.closed:
            frame = await self.frames.next(after=last, captured_after=self.next_capture_after)
            self.next_capture_after = None
            last = frame.frame_id
            if self.actions.intermission_active or frame.episode_id != self.actions.episode_id:
                continue
            async with self.decision_lock:
                if self.actions.intermission_active or frame.episode_id != self.actions.episode_id:
                    continue
                text, pictures = self._observation(frame)
                epoch = self.actions.epoch
                messages = self.session_context.model_messages(self.system, text, pictures)
                estimate = self.session_context.estimate(messages, self.tool_specs, max_output=4096)
                if estimate >= self.session_context.context_limit - 6000:
                    self.session_context.schedule_if_needed(self.client, system=self.system, tools=self.tool_specs,
                                                            output_tokens=4096, role='main')
                    if self.session_context.task:
                        await self.session_context.task
                    continue
                response = await self._ask_main(messages, self.tool_specs, frame)
                replies = []
                calls = response.get('tool_calls') or []
                for call in calls:
                    fn = call['function']
                    args = {}
                    try:
                        args = json.loads(fn['arguments']) if isinstance(fn['arguments'], str) else fn['arguments']
                        if frame.episode_id != self.actions.episode_id or self.actions.intermission_active:
                            answer = {'status': 'not_executed', 'reason': 'native_episode_ended', 'next': 'Inspect the new ready screen.'}
                        else:
                            answer = await self._tool(fn['name'], args, frame, epoch)
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        answer = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
                    self.events.add('main_tool', tool=fn['name'], arguments=args,
                                    result=answer, frame_id=frame.frame_id, episode_id=frame.episode_id)
                    replies.append({'role': 'tool', 'tool_call_id': call['id'],
                                    'content': json.dumps(answer, ensure_ascii=False, default=str)})
                self.session_context.append(text, pictures, response, replies)
                if not calls:
                    self.events.add('main_no_tool', frame_id=frame.frame_id, episode_id=frame.episode_id)
                self.session_context.schedule_if_needed(self.client, system=self.system, tools=self.tool_specs,
                                                        output_tokens=4096, role='main')



import statistics
from runtime import Agent, Evaluator
from ..backends.model_client import ModelClient


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": max(values),
    }


class BaseCoordinator:
    """Base runtime coordinator managing GameWorld environment, model client, and event logs."""

    def __init__(
        self,
        env,
        config,
        *,
        endpoint: str,
        model: str,
        output: Path,
        fps: float = 15.0,
        instruction: str = "",
        main_effort: str = "low",
        keep_playing: bool = False,
        tokenizer_file: Path | None = None,
        intermission_seconds: float = 0.0,
        screenshot_every: int = 5,
        main_compaction_threshold: int = DEFAULT_LONG_CONTEXT_TRIGGER_TOKENS,
        single_attempt: bool = False,
        api_backend: str = "openai",
        codex_config: str | Path | None = None,
        codex_auth: str | Path | None = None,
        api_slot_dir: str | Path | None = None,
        api_slot_limit: int = 2,
        **_unused_kwargs,
    ) -> None:
        self.env, self.config = env, config
        self.role_profile = build_gameworld_role_profile(config)
        library = Path(output) / "library"
        library.mkdir(parents=True, exist_ok=True)
        self.output, self.library_path = Path(output), library
        self.fps = fps
        self.intermission_seconds = intermission_seconds
        self.screenshot_every = screenshot_every
        self.instruction, self.main_effort = instruction, main_effort
        self.main_compaction_threshold = main_compaction_threshold
        self.tokenizer_file = tokenizer_file
        self.keep_playing = keep_playing
        self.single_attempt = single_attempt
        self.experiment_episode_limit = getattr(config, "experiment_episode_limit", None)
        self.events = Events(self.output)
        self.model = ModelClient(
            endpoint,
            model,
            self.events,
            self.output,
            backend=api_backend,
            codex_config=codex_config,
            codex_auth=codex_auth,
            api_slot_dir=api_slot_dir,
            api_slot_limit=api_slot_limit,
        )
        self.evaluator = Evaluator(config)
        self.agent = Agent("main", "computer_use", self.model, controls=self.role_profile.physical)
        self.stop = asyncio.Event()
        self.main = None
        self.frames = None
        self.actions = None
        self.engine = None
        self.evaluation_file = (self.output / "evaluations.jsonl").open("a", encoding="utf-8", buffering=1)


class VisualReflexCoordinator(BaseCoordinator):
    reflex_mode = 'full'
    reflex_endpoint = 'http://127.0.0.1:8189/v1/chat/completions'
    reflex_model_name = 'default'
    baseline_style = 'vista'
    # The executable AgenticRL entrypoint selects the author's VISTA runtime
    # adapter here. The old main is kept only for historical prompt export.
    main_class = VisualReflexMain

    async def _native_results(self):
        completed = 0
        episode_began = time.monotonic()
        while not self.stop.is_set():
            await asyncio.sleep(.2)
            completed_steps = self.actions.episode_completed_steps
            snapshot = await self.env.capture_state()
            if snapshot is None:
                raise RuntimeError('Official GameWorld state capture failed')
            self.frames.set_phase(snapshot.state['status'])
            await self.actions.set_scene_phase(snapshot.state['status'])
            self.agent.step_index = completed_steps
            outcome = await self.evaluator.evaluate(self.agent, snapshot.state)
            if outcome.status == 'error':
                raise RuntimeError(f'Official GameWorld evaluation failed: {outcome.metrics}')
            record = {'time': time.time(), 'monotonic': time.monotonic(),
                      'episode_id': self.frames.episode_id, 'step': self.agent.step_index,
                      'status': outcome.status, 'metrics': dict(outcome.metrics),
                      'stop_reason': outcome.stop_reason, 'phase': snapshot.state['status'],
                      'evaluation_kind': 'native_result', 'should_stop': outcome.should_stop,
                      'should_reset': outcome.should_reset}
            self.evaluation_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            if not (outcome.should_reset or outcome.should_stop):
                continue
            self.events.add('episode_result', **record)
            completed += 1
            continuous = getattr(self.main, 'continuous_episodes', False)
            final = (self.single_attempt or
                     (self.experiment_episode_limit is not None and completed >= self.experiment_episode_limit)
                     or not (outcome.should_reset or self.keep_playing))
            # Snapshot active reflex programs before intermission cleanup.
            boundary_programs = copy.deepcopy(self.engine.list()) if continuous else None
            boundary_revision = self.engine.revision
            status = ('success' if outcome.status == 'success' else
                      'budget_exhausted' if outcome.stop_reason == 'max_steps_exhausted'
                      else 'failure')
            episode_summary = {
                'episode_index': completed, 'episode_id': record['episode_id'],
                'seed': self.config.random_seed,
                'seed_supported': getattr(self.env, 'seed_supported', None),
                'applied_seed': getattr(self.env, 'seed_applied', None),
                'status': status, 'native_status': outcome.status,
                'stop_reason': outcome.stop_reason,
                'steps': self.actions.episode_steps,
                'completed_steps': self.actions.episode_completed_steps,
                'metrics': dict(outcome.metrics),
                'wall_seconds': time.monotonic() - episode_began,
            }
            self.episode_summaries.append(episode_summary)
            episode_folder = self.output / f'episode_{completed:02d}'
            episode_folder.mkdir(exist_ok=True)
            (episode_folder / 'summary.json').write_text(
                json.dumps(episode_summary, ensure_ascii=False, indent=2))
            await self.actions.begin_intermission(0)
            await self.engine.wait_pending()
            await self.engine.wait_effects()
            recent = [f for f in self.frames.ring if f.episode_id == record['episode_id'] and f.phase == 'playing'][-5:]
            if not recent:
                recent = list(self.frames.ring)[-5:]
            evidence = []
            folder = self.output / 'terminal_clips' / f'{completed:03d}'
            folder.mkdir(parents=True, exist_ok=True)
            for f in recent:
                self.frames.save(f, reason='preterminal_clip')
                path = folder / f'{f.frame_id:07d}.png'
                path.write_bytes(annotated_view(f.image))
                evidence.append((str(path), f'PREVIOUS ROUND last-five frame {f.frame_id}; t={f.timestamp:.3f}; original pixel ticks'))
            review = self.engine.after_episode(record)
            for hit in review.get('recent_reflex_previews', []):
                evidence.append((hit['preview_path'],
                                 f"PREVIOUS ROUND matched boxes={hit['hit_boxes']}; "
                                 f"GPU0 chose {hit['output']} for {hit['rule']} at frame {hit['frame_id']}; "
                                 f"inference={hit['inference_s']:.3f}s; "
                                 f"status={hit['status']}; reason={hit['reason']}; "
                                 f"executed={hit['executed']}; completed={hit['action_completed']}"))
            observation = recent[-1] if recent else self.frames.latest
            if continuous:
                if observation is not None:
                    evidence.append((str(self.frames.save(observation, reason='episode_boundary')),
                                     f'PREVIOUS ROUND raw frame {observation.frame_id}; t={observation.timestamp:.3f}'))
                review['intermission'] = {'learning': 'normal_main_loop'}
            elif self.reflex_mode == 'pure-vlm' and self.baseline_style == 'react':
                # Plain ReAct does not receive an extra episodic learning turn.
                review['intermission'] = {'turns': 0, 'card_changes': 0,
                                          'skipped_for_method': 'react'}
            else:
                review['intermission'] = await self.main.review_episode(
                    record, evidence, review, observation)
            detail = review
            if continuous:
                public_record = {key: record[key] for key in ('episode_id', 'status', 'stop_reason')}
                public_record['objective_results'] = record.get('metrics', {}).get('objective_target_results', [])
                detail = {**review, 'record': public_record, 'visual_evidence': evidence,
                          'reflex_library_revision_at_boundary': boundary_revision,
                          'reflex_programs_at_boundary': boundary_programs,
                          'program_state_timing': 'Last playing computation for each program; not an exact failure-frame snapshot. Later action receipts remain separate historical events.'}
            review_path = folder / 'review.json'
            review_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2, default=str))
            self.events.add('reflex_round_review', **review, frame_paths=[path for path, _ in evidence])
            if continuous:
                self.engine._notify('native_episode_complete', episode_id=record['episode_id'],
                                    review_path=str(review_path), final=final)
            if final:
                if continuous:
                    await self.stop.wait()
                else:
                    self.stop.set()
                return
            if continuous:
                await self.main.turn_complete.wait()
            await self.frames.suspend_for_reset()
            try:
                await self.frames.detach_page()
                if not await self.env.reset_game():
                    raise RuntimeError('Official game reset failed')
                page = self.env._require_game_manager().page
                self.actions.bind_page(page, self.role_profile.physical)
                await self.frames.bind_page(page)
                if self.config.manual_start:
                    await self.env.pause_game()
                self.frames.set_episode(self.env._episode_id)
                self.actions.set_episode(self.env._episode_id)
            finally:
                self.frames.resume_after_reset()
            await self.actions.end_intermission()
            self.agent.eval_metrics = self.evaluator.reset_metrics(self.agent.eval_metrics)
            self.agent.step_index = 0
            episode_began = time.monotonic()
            self.events.add('episode_reset', episode_id=self.frames.episode_id,
                            seed=self.config.random_seed,
                            controlled_start=bool(self.config.manual_start))

    async def run(self):
        tasks = []
        began = time.monotonic()
        status = 'completed'
        failure = None
        self.engine = None
        self.episode_summaries = []
        try:
            await self.env.start()
            if self.env.config.pause_during_inference or self.config.manual_start:
                # Pause initial scene for main-agent inspection when configured.
                await self.env.pause_game()
            page = self.env._require_game_manager().page
            self.actions = ActionArbiter(
                page, self.role_profile.physical, self.events,
                game_env=self.env, max_steps=self.config.max_steps)
            self.actions.set_episode(self.env._episode_id)
            self.frames = FrameStream(self.env, self.events, self.fps, screenshot_every=self.screenshot_every)
            tasks.append(asyncio.create_task(self.frames.run(), name='frames'))
            await self.frames.next()
            self.engine = VisualReflexEngine(self.frames, self.actions, self.events,
                                             self.role_profile.semantic, self.role_profile.specs,
                                             self.library_path, endpoint=self.reflex_endpoint,
                                             model=self.reflex_model_name, mode=self.reflex_mode,
                                             bank_enabled=self.reflex_mode not in {'no-bank', 'pure-vlm'})
            self.main = self.main_class(self.model, self.frames, self.actions, self.engine,
                                         self.events, self.role_profile, self.instruction,
                                         self.tokenizer_file, self.main_effort, self.main_compaction_threshold,
                                         game_env=self.env, baseline_style=self.baseline_style)
            self.events.add('runtime_started', model=self.model.model, endpoint=self.model.endpoint,
                            reflex_model=self.reflex_model_name, reflex_endpoint=self.reflex_endpoint,
                            reflex_mode=self.reflex_mode, game=self.config.game_id, task=self.config.task_id,
                            fps_target=self.fps, manual_start=bool(getattr(self.config, 'manual_start', False)),
                            online_apprentice=False, coding_subagent=False, automatic_observations=True,
                            baseline_style=self.baseline_style,
                            pause_during_inference=self.env.config.pause_during_inference)
            tasks += [asyncio.create_task(self.main.run(), name='main'),
                      asyncio.create_task(self.engine.run(), name='reflexes'),
                      asyncio.create_task(self._native_results(), name='native-evaluator'),
                      asyncio.create_task(self.stop.wait(), name='native-end')]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if getattr(self.main, 'finished', False):
                self.stop.set()
            if not self.stop.is_set():
                raise RuntimeError('A runtime loop ended without a native game result')
        except asyncio.CancelledError as exc:
            status, failure = 'interrupted', exc
        except BaseException as exc:
            status, failure = 'error', exc
            self.events.add('runtime_error', error=f'{type(exc).__name__}: {exc}')
        finally:
            self.stop.set()
            if self.main:
                self.main.closed = True
            frame_tasks = [task for task in tasks if task.get_name() == 'frames']
            other_tasks = [task for task in tasks if task.get_name() != 'frames']
            for task in other_tasks:
                task.cancel()
            await asyncio.gather(*other_tasks, return_exceptions=True)
            if self.frames:
                await self.frames.stop()
            if frame_tasks:
                # Drain any in-flight screenshot capture before closing the browser.
                frame_results = await asyncio.gather(*frame_tasks, return_exceptions=True)
                for error in frame_results:
                    if isinstance(error, Exception):
                        self.events.add('runtime_cleanup_error', resource='frames', error=str(error))
                        if failure is None:
                            status, failure = 'error', error
            operations = []
            if self.engine:
                operations.append(('reflex_engine', self.engine.close))
            if self.actions:
                operations.append(('actions', lambda: self.actions.reset('runtime_closed')))
            if self.frames:
                operations.append(('frames', self.frames.stop))
            if self.main:
                operations.append(('main_history', (self.main.close
                    if hasattr(self.main, 'close') else self.main.session_context.finish)))
            operations += [('environment', self.env.close_game), ('main_model', self.model.close)]
            for name, operation in operations:
                try:
                    await operation()
                except Exception as exc:
                    self.events.add('runtime_cleanup_error', resource=name, error=str(exc))
                    if failure is None:
                        status, failure = 'error', exc
            rows = self.events.rows
            summary = {'status': status, 'wall_seconds': time.monotonic()-began,
                       'model': self.model.model, 'reflex_mode': self.reflex_mode,
                       'game': self.config.game_id, 'task': self.config.task_id,
                       'episodes': self.episode_summaries,
                       'episode_count': len(self.episode_summaries),
                       'native_episodes': [r for r in rows if r['event'] == 'episode_result'],
                       'main_actions': sum(r['event'] == 'action_started' and r['source'] == 'main' for r in rows),
                       'rule_actions': sum(r['event'] == 'action_started' and r['source'].startswith('rule:') for r in rows),
                       'registered_reflexes': sum(r['event'] == 'reflex_registered' for r in rows),
                       'reflex_errors': [r for r in rows if r['event'] == 'reflex_error'],
                       'stale_main_actions_rejected': sum(r['event'] == 'main_stale_action_rejected' or
                           (r['event'] == 'action_rejected' and r.get('reason') == 'stale_main_epoch') for r in rows),
                       'champion_restores': sum(r['event'] == 'reflex_champion_restored' for r in rows),
                       'model_seconds': distribution([r['elapsed_s'] for r in rows if r['event'] == 'model_response' and r['role'] == 'main']),
                       'reflex_seconds': distribution([r['inference_s'] for r in rows if r['event'] == 'reflex_response']),
                       'none_count': sum(r['event'] == 'reflex_response' and r['output'] == 'NONE' for r in rows),
                       'escalate_count': sum(r['event'] == 'reflex_escalated' for r in rows)}
            summary_path = self.output / 'summary.json'
            summary_path.with_suffix('.tmp').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            summary_path.with_suffix('.tmp').replace(summary_path)
            self.evaluation_file.close()
            self.events.close()
        if failure:
            raise failure
        return summary
