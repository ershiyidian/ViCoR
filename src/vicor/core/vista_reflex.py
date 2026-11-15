"""ViCoR main controller and VISTA-compatible visual memory/tool runtime."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from catalog import load_game, load_task

from vista.backends.binding import recovery_prompts
from vista.backends.lifecycle_constants import (
    CHECKPOINT_PROMPT,
    MAX_CHECKPOINT_ATTEMPTS,
    compact_thresholds_for_context,
)
from vista.benchmarks.gameworld.profiles import action_specs, make_experiment, make_profile
from vista.core.contracts import (
    Capability,
    Effect,
    EvidenceKind,
    Observation,
    ObservationBundle,
    PublicTask,
    SubmissionSpec,
    ToolBinding,
    ToolReply,
    ToolSpec,
    Visual,
)
from vista.core.profiles import Phase
from vista.core.runtime import Status, TaskRuntime

from ..backends.conversation import (
    ChainConversation,
    decode_arguments,
    image_content,
    model_tools,
    project_tool_schemas,
)
from .coordinator import VisualReflexCoordinator
from .visual_reflex import TRIGGER_CONTRACT

PROMPT_EXTENSION_FILE = Path(__file__).resolve().parent / "prompts/reflex_rules.md"
READY_STATUS = 'The native game is ready for your first keyboard or mouse input.'
PROGRAM_NAME = 'visual_program'


def frame_evidence(frame, *, kind=EvidenceKind.CURRENT):
    if frame is None or not isinstance(frame.image, bytes):
        raise ValueError('VISTA needs a real current GameWorld screenshot')
    with Image.open(io.BytesIO(frame.image)) as im:
        width, height = im.size
    identity = f'episode-{frame.episode_id}-frame-{frame.frame_id}'
    return ObservationBundle((
        Observation(identity, (Visual('screen', identity, frame.image, width, height),),kind),
    ))


def reflex_request_observation(receipt, moment):
    """Register the original request screen as a historical source."""
    frame_id = (receipt['request_frame_id'] if 'request_frame_id' in receipt
                else receipt['trigger_frame_id'])
    source_id = f'episode-{receipt["episode_id"]}-frame-{frame_id}'
    pixels = Path(receipt['request_frame_path']).read_bytes()
    with Image.open(io.BytesIO(pixels)) as image:
        width, height = image.size
    return Observation(moment, (Visual('screen', source_id, pixels, width, height),),
                       EvidenceKind.HISTORICAL)


def bind_reflex_images(context, selections):
    """Bind exact crops of this chain's archived observations to a program."""
    if not isinstance(selections, dict):
        raise ValueError('images must map names to visual references')
    images = {}
    for name, selection in selections.items():
        if "event" not in selection:
            event, moment = context._reference(context.current.observations[0])
            selection = {"event": event, "moment": moment, **selection}
        reference, visual = context._select(selection)
        number, _ = context._sources[visual.source_id]
        # Reuse VISTA's original-pixel region semantics, without inspect's enlargement.
        x, y, width, height = context._with_view_size(
            {'turn': number}, 0,
            lambda _, __: context._inspection_region(selection.get('region')))
        with Image.open(io.BytesIO(visual.data)) as source:
            crop = source.convert('RGB').crop((x, y, x + width, y + height))
        encoded = io.BytesIO()
        crop.save(encoded, format='PNG')
        images[name] = {**reference,
                       'region': {'x': x, 'y': y, 'width': width, 'height': height},
                       'width': width, 'height': height,
                       'png_base64': base64.b64encode(encoded.getvalue()).decode('ascii')}
    return images


def tool_specs(engine_mode):
    word={'type':'string'}
    registration_properties={
        'images':{'type':'object',
                  'description':'Bind named RGB crops from your current observation or visual history. Regions use original-image pixels. Python reads api.images[name] for matching or tracking, and the fast visual model sees the same examples. Omit to retain existing images; a supplied mapping replaces them.',
                  'additionalProperties':{
                      'type':'object','additionalProperties':False,
                      'properties':{
                          'event':{'type':'integer','minimum':0,
                                   'description':'Source event; omit for the image in your current observation.'},
                          'moment':{'type':'integer','minimum':0,
                                    'description':'Observation within the event; omit for the final observation.'},
                          'view':{'type':'string',
                                  'description':'View name; required only for an observation with multiple views.'},
                          'region':{'type':'object','additionalProperties':False,
                                    'properties':{'x':{'type':'integer','minimum':0},
                                                  'y':{'type':'integer','minimum':0},
                                                  'width':{'type':'integer','minimum':1},
                                                  'height':{'type':'integer','minimum':1}},
                                    'required':['x','y','width','height']}},
                      'required':[]}},
        'trigger_py':{
            'type':'string',
            'description':TRIGGER_CONTRACT,
        },
        'edits':{'type':'array',
                 'description':'Apply exact old/new text replacements to the current program in order. Each old fragment identifies one occurrence.',
                 'items':{'type':'object','additionalProperties':False,
                          'properties':{'old':word,'new':word},'required':['old','new']}},
    }
    names=[
      ('set_reflex', 'Set or update a reflex rule for fast, repetitive visual reactions during active gameplay (`await api.run_rules(rules)`). Bind reference crops from an observed gameplay frame in `images` and use `"prompt"` to describe the action when a target is in range. Omitted code and images are retained.',
       registration_properties,[]),
      ('drop_reflex', 'Remove the active reflex rule and return to direct control.',{},[]),
      ('list_reflexes','Let the game run for `seconds` while the active reflex operates, then observe the screen and recent reflex actions. Use 0 for an immediate view.',
       {'seconds':{'type':'number','minimum':0,'description':'Time to observe the running reflex in seconds.'}},['seconds']),
    ]
    result=[]
    for name,desc,properties,required in names:
        result.append(ToolSpec(name,desc,{'type':'object','additionalProperties':False,
                       'properties':properties,'required':required},
                        Capability.ANNOTATION,Effect.ANNOTATION,
                         max_images=1 if name=='list_reflexes' else 0))
    return result



def _rescue_xml_tool_calls(response):
    import json, re, uuid
    calls = response.get("tool_calls") or []
    if calls:
        return calls
    content = str(response.get("content") or "")
    if "<tool_call>" not in content and "<function=" not in content:
        return calls
    act_subtypes = {"click", "click_hold", "mouse_move", "drag", "scroll", "press_key", "press_keys", "type", "wait"}
    rescued = []
    for m in re.finditer(r"<function=([a-zA-Z0-9_]+)>(.*?)</function>", content, flags=re.DOTALL):
        fn_name = m.group(1).strip()
        body = m.group(2)
        params = {}
        for pm in re.finditer(r"<parameter=([a-zA-Z0-9_]+)>\s*(.*?)\s*</parameter>", body, flags=re.DOTALL):
            k = pm.group(1).strip()
            v_raw = pm.group(2).strip()
            try:
                v = json.loads(v_raw)
            except Exception:
                v = v_raw
            params[k] = v
        if fn_name in act_subtypes:
            params.setdefault("action", fn_name)
            fn_name = "act"
        if fn_name == "act":
            for num_k in ("x", "y", "x1", "y1", "x2", "y2", "from_x", "from_y", "delta_x", "delta_y", "duration"):
                if num_k in params and isinstance(params[num_k], str):
                    try:
                        params[num_k] = float(params[num_k]) if "." in params[num_k] else int(params[num_k])
                    except Exception:
                        pass
            if params.get("action") in {"click", "click_hold"} and "button" not in params:
                params["button"] = "left"
        rescued.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": json.dumps(params)},
        })
    if rescued:
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
        response["content"] = cleaned or None
        response["tool_calls"] = rescued
    return rescued


def summarize_reflex_activity(notices):
    """Group equivalent feedback while keeping each kind of observed outcome."""
    grouped = {}
    fields = ('episode_id', 'rule', 'version', 'event', 'status', 'output',
              'reason', 'error', 'memory_40w')
    for notice in notices:
        key = tuple(notice.get(field) for field in fields) + (
            json.dumps(notice.get('action'), sort_keys=True, default=str),)
        group = grouped.setdefault(key, {
            'count': 0, 'first_history_event': notice.get('history_event')})
        group['count'] += 1
        group['latest'] = notice
    return list(grouped.values())


class VISTAReflexSession:
    """Author TaskRuntime TaskSession backed by one existing GameWorld arbiter."""

    def __init__(self, main, game, task, frame, action_spec, reflex_specs):
        self.main = main
        self.task = PublicTask(
            f'{game.game_id}/{task.task_id}/role-0', task.task_prompt,
            {'game_id':game.game_id,'task_id':task.task_id,'role':game.game_roles[0].name},
            frame_evidence(frame),
        )
        self.latest = self.task.observation
        self.submission = SubmissionSpec(Effect.ENVIRONMENT,max_images=1,
                                         costs={'game_decisions':1})
        bindings=[ToolBinding(action_spec,lambda args: self._sync_act(dict(args)))]
        for spec in reflex_specs:
            bindings.append(ToolBinding(spec,lambda args,name=spec.name: self._sync_reflex(name,args)))
        self.tools=tuple(bindings)

    def observe(self, observation):
        self.latest=observation

    def _call_on_host_loop(self,coro):
        return asyncio.run_coroutine_threadsafe(coro,self.main.host_loop).result()

    def _sync_act(self,args):
        return self._call_on_host_loop(self.main._native_action(args))

    def _sync_reflex(self,name,args):
        return self._call_on_host_loop(self.main._reflex_tool(name,args))

    def submit(self,answer):
        if self.main.final_episode_id is not None:
            return ToolReply('Native attempts are complete.',finished=True)
        return self._sync_act(None)

    def close(self):
        pass


class VISTAReflexMain:
    """Real VISTA main with *additional* conditional-reflex capabilities."""

    continuous_episodes=True

    def __init__(self, client, frames, actions, engine, events, profile, instruction,
                 tokenizer_file, effort='low', compaction_threshold=100000,
                 game_env=None, baseline_style='vista'):
        if baseline_style!='vista':
            raise ValueError('AgenticRL VISTA main uses author VISTA profile')
        self.client,self.frames,self.actions,self.engine=client,frames,actions,engine
        self.events,self.profile=events,profile
        self.env=game_env
        self.effort=effort
        self.instruction=instruction
        self.closed=False
        self.turn_complete=asyncio.Event()
        self.turn_complete.set()
        self.host_loop=asyncio.get_running_loop()
        self.runtime=None
        self.session=None
        self.messages=None
        self.conversation=ChainConversation()
        self.scene_frame=None
        self.observed_frame=None
        self.final_episode_id=None
        self.episode_history_events={}
        self.prompt_extension=PROMPT_EXTENSION_FILE.read_text(encoding='utf-8').strip()
        self.compaction_threshold=compaction_threshold
        self.game=load_game(game_env.config.game_id)
        self.task=load_task(game_env.config.game_id,game_env.config.task_id)
        if self.game.role_count != 1:
            raise ValueError('Current VISTA+reactive input arbiter is single-role; '
                             'multi-role must use VISTA role-specific sessions.')
        self.action_spec = action_specs(None, self.game, 0)[0]
        self.profile_vista=make_profile(None,self.game,self.task,0,backend='codex')
        self.reflex_specs = () if engine.mode == 'pure-vlm' else tool_specs(engine.mode)
        # Retain native runtime tools and attach reflex registration tools when reflex branch is enabled.
        if engine.mode != 'pure-vlm':
            new_tools = tuple(self.profile_vista.tools) + tuple(spec.name for spec in self.reflex_specs)
            new_prompt = replace(self.profile_vista.prompts,
                system=self.profile_vista.prompts.system + '\n\n' + self.prompt_extension)
            self.profile_vista = replace(self.profile_vista, tools=new_tools, prompts=new_prompt,
                                         profile_id='gameworld-computer-use/vista+agenticrl-reflex')
        self.events.add('vista_author_system2_selected',
            profile=self.profile_vista.profile_id,
            author_vista_base_prompt=self.profile_vista.prompts.system.split('\n\n'+self.prompt_extension)[0],
            extension_file=str(PROMPT_EXTENSION_FILE),
            author_core='TaskRuntime/TaskContext/author GameWorld action_schema')
        self.request_epoch=0
        self.request_lifecycle_epoch=0

    @property
    def finished(self):
        return (self.final_episode_id is not None or
                (self.runtime is not None and self.runtime.status==Status.COMPLETED))

    def _native_attempt_status(self):
        if self.final_episode_id is not None:
            return 'Native attempts have ended.'
        if self.actions.intermission_active:
            return 'The native attempt has ended.'
        if self.actions.awaiting_start:
            return READY_STATUS
        return 'The native attempt is active.'

    def _init_runtime(self,frame):
        self.session=VISTAReflexSession(self,self.game,self.task,frame,self.action_spec,self.reflex_specs)
        experiment=make_experiment(
            self.profile_vista,backend='codex',model=self.client.model,
            effort=self.effort,width=self.game.width,height=self.game.height,
            seed=self.env.config.random_seed)
        attempts=self.env.config.experiment_episode_limit
        steps=self.env.config.max_steps
        chain_budget=(attempts*(steps+8) if attempts is not None and steps is not None else None)
        experiment=replace(experiment,
            budgets=({'game_decisions':chain_budget,'final_answers':chain_budget}
                     if chain_budget is not None else {}),
            presentation={**dict(experiment.presentation),
                          'extension':'AgenticRL conditional-reflex tools',
                          'model_backend':'local Flash-Next',
                          'native_gameplay_clock_paused':self.env.config.pause_during_inference})
        self.runtime=TaskRuntime(self.session,experiment,
                                 artifact_dir=self.engine.directory/'vista_author_main')
        history=self.runtime._bindings['history']
        history=replace(history, spec=replace(history.spec,
            description='Browse event summaries and observations; select a single event for its complete record, including reflex code and program state.'),
            handler=self._history)
        self.runtime._bindings['history']=history
        self.runtime.specs=tuple(history.spec if spec.name=='history' else spec
                                 for spec in self.runtime.specs)
        self.tools=project_tool_schemas(model_tools(self.runtime))
        initial_text=('The game pauses between inputs and explicit waits.'
                      if self.env.config.pause_during_inference else
                      'After you start the game, it keeps running while you think and use tools.')
        initial_text += '\n' + self._native_attempt_status()
        initial_text += f'\nCurrent scene time: {frame.scene_time_s:.3f} s.'
        active_list = self.engine.list()
        if self.engine.mode != 'pure-vlm':
            initial_text += '\nCurrent active reflexes:\n' + json.dumps(
                active_list, ensure_ascii=False, default=str)
        first=self.runtime.prepare_input(Phase.INITIAL,public_text=initial_text,context_key='role-0')
        self.runtime.begin_turn(first)
        self.messages=[{'role':'system','content':self.profile_vista.prompts.system},
                       {'role':'user','content':[{'type':'text','text':first.text},
                                      *image_content(self.runtime.context.images(first.observation))]}]
        self.conversation.messages=self.messages
        self.scene_frame=frame
        self.observed_frame=frame

    def _history(self,arguments):
        reply=self.runtime.context._history_reply(arguments)
        if not reply.success:
            return reply
        content=json.loads(reply.text)
        if len(content['events'])<=1:
            return reply
        changed=False
        for event in content['events']:
            try:
                detail=json.loads(event['result'])
            except (TypeError,json.JSONDecodeError):
                continue
            if not isinstance(detail,dict):
                continue
            # Range queries render summaries while preserving full code, state, and crops in history.
            lookup={'tool':'history','arguments':{'start':event['event'],'end':event['event']+1}}
            if 'record' in detail and 'programs' in detail:
                summary={
                    'record':detail['record'],
                    'revision':detail['revision'],
                    'program_versions':{card['name']:card['version'] for card in detail['programs']},
                    'complete_record':lookup}
            elif (event.get('action') or {}).get('tool') == 'reflex' or detail.get('rule') is not None:
                summary={
                    key: detail[key] for key in (
                        'event', 'rule', 'version', 'episode_id', 'frame_id',
                        'request_frame_id', 'action', 'action_id', 'executed',
                        'status', 'reason', 'error', 'moments'
                    ) if key in detail
                }
                summary['complete_record'] = lookup
            else:
                continue
            event['result']=json.dumps(summary,ensure_ascii=False,separators=(',',':'))
            changed=True
        return replace(reply,text=json.dumps(content,ensure_ascii=False,separators=(',',':'))) if changed else reply

    def _dispatch_screenshot_after(self, original, *, received_after=None):
        async def request():
            return await self.frames.next(after=original.frame_id,captured_after=received_after)
        return request()

    async def _native_action(self,args):
        self.actions.main_busy = False
        try:
            return await self._native_action_inner(args)
        finally:
            self.actions.main_busy = True

    async def _native_action_inner(self,args):
        if self.final_episode_id is not None:
            return ToolReply('Native attempts are complete.',False)
        if args is not None and not isinstance(args,dict):
            return ToolReply('Invalid GameWorld action request.',False)
        frame=self.frames.latest
        if frame is None or self.actions.intermission_active:
            return ToolReply('Native episode is over; no input executed.',False)
        if frame.episode_id!=self.actions.episode_id:
            return ToolReply('Stale episode action discarded.',False)
        try:
            action=None if args is None else dict(args)
            receipt=await self.actions.submit(action,source='main',frame=frame,
                                              expected_epoch=self.request_epoch,
                                              expected_lifecycle_epoch=self.request_lifecycle_epoch)
            if receipt.get('status')=='no_action':
                after=await self._dispatch_screenshot_after(frame,received_after=receipt['finished_at'])
                self.scene_frame=self.observed_frame=after
                self.events.add('vista_empty_decision',episode_id=frame.episode_id,
                                action=None,output=receipt,frame_id=after.frame_id)
                return ToolReply('No valid action was executed. Final game screen.',
                                 False,frame_evidence(after))
            if receipt.get('status')=='executing':
                completed=await self.actions.wait_for_completion(receipt['action_id'])
                receipt={**receipt,'status':'completed' if completed['reason']=='duration_completed'
                         else 'interrupted','reason':completed['reason'],
                         'finished_at':completed['monotonic'],'held_s':completed['held_s']}
            ok=receipt.get('status')=='completed'
            if receipt.get('execution_started_at') is None:
                return ToolReply('No valid input executed: '+json.dumps(receipt,default=str),False)
            after=await self._dispatch_screenshot_after(frame,received_after=receipt.get('finished_at'))
            if after.episode_id!=frame.episode_id:
                return ToolReply('Native input ended before the next attempt; its outcome is in visual history: '
                                 +json.dumps(receipt,default=str),ok)
            self.scene_frame=after
            response=frame_evidence(after)
            self.events.add('vista_action_receipt',episode_id=frame.episode_id,
                            action=action,output=receipt,frame_id=after.frame_id)
            text=('Native action completed.' if ok else
                  'Native action interrupted: '+json.dumps(receipt,default=str))
            text += ' ' + self._native_attempt_status()
            text += f' Current scene time: {after.scene_time_s:.3f} s.'
            self.observed_frame=after
            return ToolReply('Current game screen. '+text,ok,response)
        except (ValueError,KeyError,TypeError) as exc:
            self.events.add('vista_invalid_action',reason=str(exc),action=args)
            return ToolReply(f'Invalid GameWorld action: {type(exc).__name__}: {exc}',False)

    async def _reflex_tool(self,name,args):
        if not isinstance(args,dict):
            return ToolReply('Invalid reflex arguments.',False)
        try:
            if name=='set_reflex':
                if 'images' in args:
                    args = {**args, 'images': bind_reflex_images(self.runtime.context, args['images'])}
                outcome=self.engine.set_reflex(PROGRAM_NAME,**args)
                event=self.runtime.context.record_event(
                    ObservationBundle(()),
                    result=json.dumps({
                        'record':outcome,'revision':self.engine.revision,
                        'programs':self.engine.list(),
                    },ensure_ascii=False,default=str),
                    update_current=False)
                return ToolReply(json.dumps({**outcome,'history_event':event},ensure_ascii=False),True)
            if name=='drop_reflex':
                result=self.engine.drop_reflex(PROGRAM_NAME)
                await self.actions.disable_source('rule:'+PROGRAM_NAME)
                return ToolReply(json.dumps(result,ensure_ascii=False),True)
            if name=='list_reflexes':
                secs = float(args.get('seconds', 0) or 0)
                if secs > 0:
                    self.actions.main_busy = False
                    try:
                        frame = self.frames.latest
                        if frame is not None and not self.actions.intermission_active:
                            async with self.actions.lock:
                                w = await self.actions._start_passive_wait(
                                    {'action': 'wait', 'duration': secs},
                                    source='reflex_observe', frame=frame)
                            await self.actions.wait_for_completion(w['action_id'])
                            after = await self.frames.capture(source='reflex_observe_after')
                            if after is not None:
                                self.scene_frame = after
                    finally:
                        self.actions.main_busy = True
                programs=self.engine.list()
                text=json.dumps(programs,ensure_ascii=False,default=str)
                receipt=programs[0]['reflex'] if len(programs)==1 else {}
                if not receipt.get('preview_path'):
                    return ToolReply(text,True)
                image=Path(receipt['preview_path']).read_bytes()
                source_evidence=ObservationBundle((reflex_request_observation(
                    receipt,'reflex-source-'+str(time.monotonic_ns())),))
                self.runtime.context.observe(source_evidence,update_current=False)
                with Image.open(io.BytesIO(image)) as im:width,height=im.size
                source=source_evidence.observations[0].views[0].source_id
                source_event,source_moment=self.runtime.context._reference(source_evidence.observations[0])
                obs=Observation('reflex-preview-'+str(time.monotonic_ns()),
                     (Visual('reflex_preview','reflex-'+str(time.monotonic_ns()),image,width,height,
                             derived_from=(source,),
                             transform={**receipt['preview_transform'],
                                        'event':source_event,'moment':source_moment,'view':'screen'}),),
                                EvidenceKind.DERIVED)
                return ToolReply(text,True,ObservationBundle((obs,)))
        except Exception as exc:
            self.events.add('vista_reflex_tool_error',tool=name,error=str(exc))
            return ToolReply(f'{type(exc).__name__}: {exc}',False)
        return ToolReply('Unknown reflex tool.',False)

    async def _ask_main(self):
        frame=self.scene_frame or self.frames.latest
        self.runtime.acknowledge()
        self.request_epoch=self.actions.epoch
        self.request_lifecycle_epoch=self.actions.lifecycle_epoch
        active_tools=self.tools
        if self.conversation.checkpoint_pending and getattr(self.conversation,'checkpoint_attempts',0)>=1:
            cp_tools=[t for t in self.tools if (t.get('function') or {}).get('name') in ('save_compact_checkpoint','read_guide','write_guide','read_working','write_working')]
            if cp_tools:
                active_tools=cp_tools
        async with self.actions.inference():
            started=time.monotonic()
            try:
                return await self.client.ask(self.messages,active_tools,role='main',
                                             max_tokens=8192,effort=('low' if self.conversation.checkpoint_pending else self.effort),
                                             reasoning_budget_tokens=({'high':2048,'medium':768,'low':256}.get(self.effort) if self.client.backend=='ninfer' else None))
            finally:
                self.events.add('vista_main_inference',elapsed_s=time.monotonic()-started,
                                episode_id=frame.episode_id if frame else None,
                                paused_game_clock=self.actions.clock_env is not None)

    def _new_user(self,task_input,*,include_image=True):
        return {'role':'user','content':[{'type':'text','text':task_input.text},
                    *(image_content(self.runtime.context.images(task_input.observation))
                      if include_image else [])]}

    def _begin_observation(self,frame,text,phase=Phase.STEP):
        self.runtime.acknowledge()
        previous=self.observed_frame
        include_image=(previous is None or previous.episode_id!=frame.episode_id
                       or previous.image!=frame.image)
        text += '\n' + self._native_attempt_status()
        text += f'\nCurrent scene time: {frame.scene_time_s:.3f} s.'
        if self.engine.cards:
            card = next(iter(self.engine.cards.values()))
            name = card['name']
            counts = self.engine.stats.get(name, {})
            decision = self.engine.state.get(name, {}).get('last_decision')
            info = {
                'version': card['version'],
                'queries': counts.get('query_count', 0),
                'actions': counts.get('action_count', 0),
                'nones': counts.get('none_count', 0),
                'last_decision': ({key: decision.get(key) for key in ('status', 'action', 'executed')} if decision else None),
            }
            mem = self.engine.state.get(name, {}).get('memory')
            if mem:
                mem_s = json.dumps(mem, ensure_ascii=False, separators=(',', ':'), default=str)
                info['state'] = mem if len(mem_s) <= 400 else (mem_s[:400] + '...')
            text += '\nActive helper rule: ' + json.dumps(info, ensure_ascii=False, separators=(',', ':'), default=str)
        new=self.runtime.prepare_input(phase,observation=frame_evidence(frame),
                                       public_text=text,context_key='role-0')
        self.runtime.begin_turn(new)
        self.messages.append(self._new_user(new,include_image=include_image))
        self.observed_frame=frame

    def _archive_reflex_outcomes(self,notices):
        """Archive reflex execution receipts in visual history and surface concise turn summaries."""
        outcomes={'reflex_visual_result','reflex_receipt','reflex_stale_result','reflex_decision','reflex_escalation',
                  'reflex_action_not_executed','reflex_effect_unavailable','reflex_effect_error'}
        for index,receipt in enumerate(notices):
            if receipt.get('event') not in outcomes:
                continue
            before_path=receipt.get('before_frame_path')
            after_path=receipt.get('after_frame_path')
            try:
                frames=[]
                moments={}
                if receipt.get('request_frame_path'):
                    moments['request']=len(frames)
                    identity=f'reflex-{receipt["episode_id"]}-notice-{receipt["id"]}-request'
                    frames.append(reflex_request_observation(receipt,identity))
                for stage,path in [('observed',receipt.get('observed_frame_path')),
                                   ('before',before_path),('after',after_path)]:
                    if not path:
                        continue
                    pixels=Path(path).read_bytes()
                    with Image.open(io.BytesIO(pixels)) as image:
                        width,height=image.size
                    identity=(f'reflex-{receipt["episode_id"]}-'
                              f'notice-{receipt["id"]}-{stage}')
                    moments[stage]=len(frames)
                    frames.append(Observation(identity,
                        (Visual('screen',identity,pixels,width,height),),
                        EvidenceKind.HISTORICAL))
                event=self.runtime.context.record_event(
                    ObservationBundle(tuple(frames)),
                    action={'tool':'reflex', 'rule':receipt['rule'],
                            'native_action':receipt.get('action'),
                            'action_id':receipt.get('action_id')},
                    result=json.dumps({**receipt,'moments':moments},
                                      ensure_ascii=False,default=str),
                    success=((receipt['event']=='reflex_visual_result'
                              and receipt.get('status') in {'confirmed','no_action'})
                             or (receipt['event']=='reflex_receipt'
                                 and receipt.get('executed') is True)),
                    update_current=False,environment=(receipt.get('executed') is True
                        or (receipt.get('action_id') is not None and receipt['event']
                            in {'reflex_effect_unavailable','reflex_effect_error'})))
                summary={key:receipt[key] for key in (
                    'event','output','action','executed','status','reason',
                    'error','observation') if key in receipt}
                if 'reason' not in summary and receipt.get('action_completed_reason'):
                    summary['reason']=receipt['action_completed_reason']
                notices[index]={**summary,'history_event':event}
                self.events.add('vista_reflex_history_recorded',
                    episode_id=receipt['episode_id'],action_id=receipt.get('action_id'),
                    rule=receipt['rule'],history_event=event,
                    notice_id=receipt['id'],notice_event=receipt['event'],
                    preview_path=receipt.get('preview_path'),moments=moments,
                    before_frame_path=before_path,after_frame_path=after_path)
            except Exception as exc:
                receipt['vista_history_error']=f'{type(exc).__name__}: {exc}'
                self.events.add('vista_reflex_history_error',
                    action_id=receipt.get('action_id'),error=receipt['vista_history_error'])

    async def _model_turn(self):
        self.turn_complete.clear()
        self.actions.main_busy = True
        if self.actions.clock_env is not None and not self.actions.clock_actions:
            await self.actions.clock_env.pause_game()
        try:
            return await self._complete_model_turn()
        finally:
            self.turn_complete.set()

    async def _complete_model_turn(self):
        runtime=self.runtime
        self.runtime.acknowledge()
        response=await self._ask_main()
        recent=next((event for event in reversed(self.events.rows)
                     if event.get('event')=='model_response' and event.get('role')=='main'),None)
        if recent:
            usage=recent.get('usage') or {}
            prompt_tokens=usage.get('prompt_tokens',0)
            if isinstance(prompt_tokens,int) and prompt_tokens>0:
                self.conversation.prompt_tokens=prompt_tokens
                runtime.observe_context_tokens(prompt_tokens)
        self.messages.append(response)
        calls=_rescue_xml_tool_calls(response)
        images=[]
        if calls:
            for call in calls:
                name=call['function']['name']
                arguments=decode_arguments(call)
                reply=await asyncio.to_thread(runtime.execute,name,arguments,call['id'])
                self.messages.append({'role':'tool','tool_call_id':call['id'],'content':reply.text})
                images.extend(reply.images)
                self.events.add('vista_main_tool',tool=name,arguments=arguments,
                                success=reply.success)
                self.runtime.acknowledge()
                if runtime.terminal:
                    break
        elif not self.conversation.checkpoint_pending:
            if recent and recent.get('finish_reason')=='length':
                self.messages.append({'role':'user','content':'Continue.'})
            else:
                reply=await asyncio.to_thread(runtime.submit,response.get('content') or '')
                self.messages.append({'role':'user','content':reply.text})
                images.extend(runtime.context.images(reply.observation))
        if images:
            self.messages.append({'role':'user','content':image_content(images)})
        if runtime.failure:
            raise RuntimeError('VISTA TaskRuntime failed: '+runtime.failure)
        self.runtime.acknowledge()
        return response, recent.get('finish_reason') if recent else None

    def _archive_episode_results(self,notices):
        for index,notice in enumerate(notices):
            if notice.get('event')!='native_episode_complete':
                continue
            # Record episode boundary in history while maintaining observation and compaction state in the main loop.
            text=Path(notice['review_path']).read_text(encoding='utf-8')
            detail=json.loads(text)
            record=detail['record']
            # Use the author's episode_summary status, including budget exhaustion.
            row={'episode_index':len(self.episode_history_events)+1,
                 'status':('success' if record['status']=='success' else
                           'budget_exhausted' if record.get('stop_reason')=='max_steps_exhausted'
                           else 'failure')}
            evidence=detail['visual_evidence']
            confirmations={item['preview_path']:item
                           for item in detail.get('recent_reflex_previews',[])
                           if item.get('preview_path')}
            historical=[]
            for i,(path,label) in enumerate(evidence):
                src=f'episode-{record["episode_id"]}-clip-{i}-{time.monotonic_ns()}'
                if path in confirmations:
                    historical.append(reflex_request_observation(confirmations[path],src))
                    continue
                data=Path(path).read_bytes()
                with Image.open(io.BytesIO(data)) as image:w,h=image.size
                historical.append(Observation(src,(Visual('screen',src,data,w,h),),
                                              EvidenceKind.HISTORICAL))
            event=self.runtime.context.record_event(
                ObservationBundle(tuple(historical)),
                result=json.dumps({
                    'record': {'episode_id': record['episode_id'], **row},
                    'revision':
                        detail['reflex_library_revision_at_boundary'],
                    'programs':
                        detail['reflex_programs_at_boundary'],
                    'program_state_timing': detail['program_state_timing'],
                }, ensure_ascii=False, default=str),
                update_current=False,environment=True)
            self.episode_history_events[record['episode_id']]=event
            notices[index]={
                'id':notice['id'],'event':notice['event'],'episode_id':record['episode_id'],
                **row,
                'history_event':event,
                'final':notice['final']}
            if notice['final']:
                self.final_episode_id=record['episode_id']
            self.events.add('vista_episode_history_recorded',
                            episode_id=record['episode_id'],history_event=event,
                            review_path=notice['review_path'],final=notice['final'])

    async def run(self):
        frame=await self.frames.next()
        self._init_runtime(frame)
        last_epis=frame.episode_id
        while not self.closed:
            if self.runtime.terminal:
                if self.finished:return
                raise RuntimeError('Author VISTA runtime exhausted before final submission')
            frame=self.frames.latest
            final_available=(self.final_episode_id is not None or any(
                notice.get('event')=='native_episode_complete' and notice.get('final')
                for notice in self.engine.notifications))
            if (frame is None or frame.episode_id!=self.actions.episode_id or
                    (self.actions.intermission_active and not final_available)):
                await asyncio.sleep(0.07)
                continue
            notices=self.engine.drain_notifications()
            if notices:
                self._archive_reflex_outcomes(notices)
                self._archive_episode_results(notices)
            if self.final_episode_id is not None:
                return
            frame=self.frames.latest
            self.scene_frame=frame
            text=[]
            if notices:
                receipts=summarize_reflex_activity(
                    notice for notice in notices if notice.get('event')!='native_episode_complete')
                if receipts:
                    text.append('Activity since your last turn (counts with latest records): '+
                                json.dumps(receipts,ensure_ascii=False,default=str))
                for notice in notices:
                    if notice.get('event')=='native_episode_complete':
                        text.append(f"Attempt {notice['episode_index']} ended with status {notice['status']}; history event {notice['history_event']}.")
                self.events.add('vista_reflex_notices_delivered',
                                frame_id=frame.frame_id,episode_id=frame.episode_id,
                                event_ids=([group['latest'].get('id') for group in receipts]
                                           + [notice.get('id') for notice in notices
                                              if notice.get('event')=='native_episode_complete']))
            if frame.episode_id!=last_epis:
                last_epis=frame.episode_id
                text.append(f'Attempt {len(self.episode_history_events)+1} has started with a fresh environment. '
                            'The task objective is unchanged. GUIDE, WORKING, conversation, '
                            'and visual history are retained.')
                if self.actions.awaiting_start:
                    text.append(READY_STATUS)
            previous=self.observed_frame
            if (text or previous is None or previous.episode_id!=frame.episode_id
                    or previous.scene_time_s!=frame.scene_time_s
                    or previous.image!=frame.image):
                self._begin_observation(frame,'\n'.join(text),
                    phase=Phase.CHECKPOINT if self.conversation.checkpoint_pending else Phase.STEP)
            last_calls = []
            if not self.conversation.checkpoint_pending:
                resp_msg, _ = await self._model_turn()
                last_calls = [c["function"]["name"] for c in (resp_msg.get("tool_calls") or []) if "function" in c]
            if not self.runtime.terminal:
                await self._check_checkpoint()
            should_auto_obs = (
                bool(self.engine.cards)
                and "act" not in last_calls
                and "list_reflexes" not in last_calls
                and not self.conversation.checkpoint_pending
                and (self.actions.clock_env is None or "set_reflex" in last_calls)
            )
            if should_auto_obs:
                self.actions.main_busy = False
                try:
                    if self.actions.clock_env is not None and not self.actions.intermission_active:
                        await self.frames.capture(source='reflex_wake')
                    steps = 45 if self.actions.clock_env is None else 15
                    for _ in range(steps):
                        if (self.closed or self.runtime.terminal or self.actions.intermission_active
                                or (self.frames.latest and self.frames.latest.episode_id != last_epis)
                                or any(n.get("event") in ("native_episode_complete", "reflex_error", "reflex_escalation")
                                       for n in self.engine.notifications)):
                            break
                        await asyncio.sleep(0.1)
                finally:
                    self.actions.main_busy = True
                    if self.actions.clock_env is not None and not self.actions.clock_actions:
                        await self.actions.clock_env.pause_game()
            else:
                await asyncio.sleep(0)

    async def _check_checkpoint(self):
        runtime=self.runtime
        current=self.conversation.prompt_tokens
        threshold=min(52000, compact_thresholds_for_context(131072,source='model_context_131k').normal_auto_compact_limit)
        if current>=threshold and not self.conversation.checkpoint_pending:
            runtime.context.request_compact_checkpoint(reason='model_context_threshold')
            self.conversation.checkpoint_pending=True
            self.conversation.checkpoint_attempts=0
        if not self.conversation.checkpoint_pending:return
        # Preserve pending checkpoint state across episode boundaries.
        while not runtime.context.compact_checkpoint_ready.exists():
            if self.actions.intermission_active and self.final_episode_id is None:
                return
            if self.conversation.checkpoint_attempts>=MAX_CHECKPOINT_ATTEMPTS:
                raise RuntimeError('VISTA failed to create compact checkpoint')
            checkpoint=runtime.prepare_input(Phase.CHECKPOINT,
                        public_text=CHECKPOINT_PROMPT,context_key='role-0')
            runtime.begin_turn(checkpoint)
            self.messages.append({'role':'user','content':checkpoint.text})
            await self._model_turn()
            self.conversation.checkpoint_attempts+=1
        recovered=runtime.context.begin_compact_recovery()
        recovery_text=recovery_prompts(runtime).compact(recovered)
        recovery_text += '\n' + self._native_attempt_status()
        recovery_text += '\nCurrent active reflexes:\n' + json.dumps(
            self.engine.list(), ensure_ascii=False, default=str)
        runtime.fresh_context('role-0')
        next_input=runtime.prepare_input(Phase.RECOVERY,public_text=recovery_text,
                   observation=runtime.current_observation,context_key='role-0')
        runtime.begin_turn(next_input)
        self.messages=[{'role':'system','content':self.profile_vista.prompts.system},
                       self._new_user(next_input)]
        self.conversation.messages=self.messages
        runtime.context.complete_compact_recovery(
            recovered,delivery='openai_compatible_fresh_context')
        self.conversation.prompt_tokens=0
        self.conversation.checkpoint_pending=False

    async def close(self):
        if self.runtime is not None:
            notices=self.engine.drain_notifications()
            self._archive_reflex_outcomes(notices)
            self._archive_episode_results(notices)
        if self.session is not None:self.session.close()


class VISTAReflexCoordinator(VisualReflexCoordinator):
    main_class=VISTAReflexMain
    baseline_style='vista'
