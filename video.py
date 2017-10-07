# -*- coding: utf-8 -*-
import imageio
imageio.plugins.ffmpeg.download()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from car.train import load_model
from car.detect import ImgDetector, VideoDetector

if __name__ == "__main__":

    detector = VideoDetector(ImgDetector(classifier=load_model("model_v4.pkl")))
    def process_image(image):
        return detector.run(image, False)
 
    white_output = 'project_video_result.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
