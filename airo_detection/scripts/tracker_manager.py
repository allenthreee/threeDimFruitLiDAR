class Track:
    def __init__(self, id):
        self.id = id
        self.tracked_counts = 0
        self.bounding_boxes = []
        self.mean_size = 0
        self.mean_lidar_depth = 0

    def update(self, bounding_box):
        self.tracked_counts += 1
        self.bounding_boxes.append(bounding_box)
        for bbx in self.bounding_boxes:
            self.mean_size = (self.mean_size + bbx.radius)/2
            # print(f"here we go: {self.mean_lidar_depth} and {bbx.lidar_d}")
            self.mean_lidar_depth = (self.mean_lidar_depth + bbx.lidar_d)/2

class TrackManager:
    def __init__(self):
        self.tracks = {}
        self.total_count = 0

    def update(self, id, bounding_box):
        if id not in self.tracks:
            self.tracks[id] = Track(id)
        self.tracks[id].update(bounding_box)

    def get_track(self, id):
        return self.tracks.get(id, None)

    def get_all_tracks(self):
        self.total_count = 0
        # print(f"\n\nget_all_tracks called=======================")
        for track_id, track in self.tracks.items():
            # print(f"\nTrack ID: {track_id}, the color is {track.bounding_boxes[0].Class}")
            # print(f"Track count is: {track.tracked_counts}")
            # print(f"Track mean_size is: {track.mean_size}")
            # print(f"Track mean_mean_lidar_depth is: {track.mean_lidar_depth}")
            self.total_count = self.total_count + track.tracked_counts
        # print(f"total count is: {self.total_count}")
        
        # print(f"the {track_id}th fruit track\n\n")
        # for id in range(1, len(self.tracks)):
        #     print(f"id of {id}th track is: {self.tracks[id].id}")
        #     print(f"frames of {id}th track is: {self.tracks[id].frames_tracked}")
        #     print(f"bbxes of {id}th track is: {self.tracks[id].bounding_boxes}\n")
        return self.tracks.values()
# Usage
# track = Track(id=1)
# for frame in video_frames:
#     bounding_box = detect_object(frame)
#     track.update(bounding_box)

# print(f"Object {track.id} has been tracked for {track.frames_tracked} frames.")