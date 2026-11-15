import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..backends.model_client import image_part, main_visual_view


class VisualArchive:
    def __init__(self, events, directory: Path):
        self.events = events
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending_images = []

    def frames(self):
        frames = {row['frame_id']: dict(row) for row in self.events.rows if row['event'] == 'frame'}
        for row in self.events.rows:
            if row['event'] == 'frame_saved' and row['frame_id'] in frames:
                frames[row['frame_id']]['saved_path'] = row['path']
        return {number: row for number, row in frames.items() if row.get('saved_path')}

    def inspect(self, frame_id: int, region: dict | None, question: str):
        record = self.frames()[frame_id]
        picture = Image.open(record['saved_path']).convert('RGB')
        original_size = list(picture.size)
        scale = 1
        if region is not None:
            x, y, width, height = (int(region[key]) for key in ('x', 'y', 'width', 'height'))
            if not (0 <= x < x + width <= picture.width and 0 <= y < y + height <= picture.height):
                raise ValueError('Crop region exceeds source image bounds')
            bounds = (x, y, x + width, y + height)
            picture = picture.crop(bounds)
            scale = min(4, max(1, 640 // max(picture.size)))
            picture = picture.resize((picture.width * scale, picture.height * scale), Image.Resampling.NEAREST)
        else:
            # Historical frames retain original viewport coordinates unless explicitly cropped.
            view, bounds = main_visual_view(Path(record['saved_path']).read_bytes())
            picture = Image.open(BytesIO(view)).convert('RGB')
        output = BytesIO()
        picture.save(output, format='PNG')
        item = {'frame_id': frame_id, 'episode_id': record['episode_id'],
                'question': question, 'region': region,
                'original_size': original_size,
                'display_bounds_in_original': list(bounds), 'display_scale': scale,
                'coordinate_mapping': 'Original x/y = displayed x/y divided by display_scale, plus the bounds left/top.'}
        # Frame indices are not fixed-interval ticks; expose capture timestamps and inputs.
        captured_at = record['captured_at']
        same_episode = [row for row in self.events.rows if row.get('episode_id') == record['episode_id']]
        origin = min(row['captured_at'] for row in same_episode if row['event'] == 'frame')
        prior_inputs = [row for row in same_episode if row['event'] == 'action_started'
                        and row['monotonic'] <= captured_at]
        item.update(captured_at_monotonic_s=captured_at, phase=record.get('phase'),
                    episode_elapsed_wall_ms=round((captured_at-origin)*1000, 3))
        if prior_inputs:
            preceding = prior_inputs[-1]
            item['preceding_actual_input'] = {
                **{key: preceding[key] for key in ('action_id', 'source', 'action')},
                'elapsed_wall_ms': round((captured_at-preceding['monotonic'])*1000, 3),
            }
        self.pending_images.append((item, output.getvalue()))
        self.pending_images = self.pending_images[-4:]
        self.events.add('visual_archive_inspected', **item)
        return {**item, 'status': 'image_attached_to_next_request'}

    def history(self, limit: int = 12):
        count = min(max(int(limit), 1), 32)
        events = [row for row in self.events.rows
                  if row['event'] in {'action_started', 'episode_result', 'rule_wake_requested', 'fast_decision_result'}]
        records = []
        for row in events[-count:]:
            item = {key: row[key] for key in ('event', 'episode_id', 'frame_id', 'action_id', 'source', 'action', 'status') if key in row}
            if row['event'] == 'episode_result':
                item['objectives'] = [{key: objective[key] for key in ('field', 'actual', 'target')}
                                      for objective in row['metrics']['objective_target_results']]
            records.append(item)
        frames = list(self.frames().values())
        selected = [frames[index] for index in sorted({0, max(0, len(frames)-1), max(0, len(frames)//2)})] if frames else []
        return {'records': records, 'available_frames': [
            {'frame_id': row['frame_id'], 'episode_id': row['episode_id'], 'phase': row.get('phase')} for row in selected]}

    def write_note(self, name: str, content: str):
        if name not in {'GUIDE.md', 'WORKING.md'}:
            raise ValueError('Only GUIDE.md or WORKING.md may be written')
        if len(content) > 2400:
            raise ValueError('Each note is limited to 2400 characters; compact before saving')
        (self.directory / name).write_text(content, encoding='utf-8')
        self.events.add('vista_note_written', name=name, content=content)
        return {'status': 'saved', 'name': name}

    def notes(self):
        return {name: (self.directory / name).read_text(encoding='utf-8')
                for name in ('GUIDE.md', 'WORKING.md') if (self.directory / name).is_file()}


