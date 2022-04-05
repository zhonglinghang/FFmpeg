
class FrameProcessor:
    delay = 5
    queue = []
    count = 0

    def __init__(self, d):
        self.delay = d
    
    def process_frame(self, frame):
        self.queue.append(frame)
        self.count += 1
        if self.count < self.delay:
            return []
        return [self.queue.pop(0)]
    def flush_frames(self):
        return self.queue

def query_formats():
    return ['yuva420p']

def setup(w, h, pixfmt, depth=5):
    print('setup(), delay={}'.format(depth))
    return {
        'config': {
            'w' : w,
            'h' : h,
            'pixfmt': pixfmt,
            'process_mode':'one_to_many',
        },
        'processor': FrameProcessor(depth)
    }

def process_frame(frame, processor):
    print('FrameProcessor.process_frame() {}'.format(frame.pts))
    return processor.process_frame(frame)

def flush_frames(processor):
    print('FrameProcessor.flush_frames()')
    return processor.flush_frames()