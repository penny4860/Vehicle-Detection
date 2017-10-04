# -*- coding: utf-8 -*-
import imageio
from car.train import load_model
from car.detect import ImgDetector

imageio.plugins.ffmpeg.download()


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# from detector.framework import ImageFramework

def process_image(image):
    d = ImgDetector(classifier=load_model("model_hnm.pkl"))
    img_draw = d.run(image, do_heat_map=True)
    return img_draw
 
if __name__ == "__main__":
    white_output = 'test_video_result.mp4'
    clip1 = VideoFileClip("test_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    print("done")
