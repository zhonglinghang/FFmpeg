
def query_formats():
    return ['yuv420p', 'yuv422p']

def setup(w, h, pixfmt):
    return {
        'config': {
            'w' : w,
            'h' : h,
            'pixfmt': pixfmt,
            'fr_ratio':2,
            'process_mode':'one_to_many'
        }
    }

def process_frame(frame):
    frame2 = frame.clone()
    frame2.pts += 10
    # print('process_frame() {}  2: {}'.format(frame.pts, frame2.pts))
    return [frame, frame2]

def flush_frames():
    return []