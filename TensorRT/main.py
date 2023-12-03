from rt.models.torch_utils import det_postprocess
from rt.models.cudart_api import TRTEngine
from rt.models.utils import blob
from ultralytics import YOLO
import numpy as np
import os
import telebot
import cv2
import torch

def ImageBox(image, new_shape=(640, 640), color=(0, 0, 0)):

    width, height, channel = image.shape

    ratio = min(new_shape[0] / width, new_shape[1] / height)

    new_unpad = int(round(height * ratio)), int(round(width * ratio))

    dw, dh = (new_shape[0] - new_unpad[0])/2, (new_shape[1] - new_unpad[1])/2

    if (height, width) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, ratio, (dw, dh)# Initialize your bot token

#TOKEN = '6251722523:AAEI7a-GW4dRneTM8LSnRgr-1swvpaAOAF0'
TOKEN = "6838811403:AAGUJFUwqR52m0O7u77fcH0L5Z0pk1ICdrM"
bot = telebot.TeleBot(TOKEN)

exec_path = os.getcwd()
model_path = os.path.join(exec_path, "/drive/MyDrive/TensorRT/thebest.engine")
print(model_path)
enggine = TRTEngine(model_path)




@bot.message_handler(content_types=['text'])
def handle_message(message):

    mess = f"Salom 10Mb gacha bo'lgan videoni jo'nating"
    bot.send_message(message.chat.id, mess)

# Define the folder where videos will be saved
VIDEO_FOLDER = 'videos'

@bot.message_handler(content_types=['video'])
def handle_video(message):
    try:
        video_file = message.video
        print(video_file.file_size)
        if video_file.file_size > 10 * 1024 * 1024:  # 10MB in bytes
            bot.reply_to(message, "It's too big! Please send a smaller video.(10MB)")
        else:
            # Process the video or perform any other desired action
            bot.reply_to(message, "Video received. Thank you!")

        # Create the video folder if it doesn't exist
        if not os.path.exists(VIDEO_FOLDER):
            os.makedirs(VIDEO_FOLDER)

        # Get the file info
        file_info = bot.get_file(message.video.file_id)
        file_path = file_info.file_path

        # Download the video
        downloaded_file = bot.download_file(file_path)
        msg=bot.reply_to(message, "video downloaded successfully")
        # Save the video to the folder
        video_path = os.path.join(VIDEO_FOLDER, f"{message.video.file_id}.avi")
        output_file = os.path.join(VIDEO_FOLDER, f"out{message.video.file_id}.avi")

        with open(video_path, 'wb') as new_file:
            new_file.write(downloaded_file)


        H, W = enggine.inp_info[0].shape[-2:]
        cap = cv2.VideoCapture(video_path)
        imageWidth = int(cap.get(3))
        imageHeight = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        freyms = int(cap.get(7))
        #print("fourcc=",cap.get(cv2.CAP_PROP_BITRATE))
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        out = cv2.VideoWriter(output_file, fourcc, fps, (imageWidth, imageHeight))
        k=0

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if k%2==0:
              k+=1
              continue
            k+=1
            try:
                if int(k/freyms*100)%10==0:
                    bot.edit_message_text(f"{int(k/freyms*100)}%completed...",msg.chat.id,msg.message_id)
            except :
                pass

            image, ratio, dwdh = ImageBox(frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            tensor = blob(image, return_seg=False)
            tensor = torch.asarray(tensor)

            results = enggine(tensor)

            dwdh = np.array(dwdh * 2, dtype=np.float32)

            bboxes, scores, labels = det_postprocess(results)
            bboxes = (bboxes-dwdh)/ratio


            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().astype(np.int32).tolist()

                cv2.rectangle(frame, (bbox[0],bbox[1]) , (bbox[2],bbox[3]) , (255,255,255), 1)

                cv2.putText(frame,
                            f'{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)

            out.write(frame)

        bot.edit_message_text(f"the process is completed", msg.chat.id, msg.message_id)
        cap.release()
        out.release()
        # Send a confirmation message
        #res = bot.reply_to(message, "Loading result...")

        # Send the video back to the user
        try:
            with open(output_file, 'rb') as video_file:
                bot.send_video(message.chat.id, video_file)
        except Exception as er:
            bot.reply_to(message, f"{str(er)}")
        # Delete the video file
        os.remove(video_path)

        os.remove(output_file)


    except Exception as e:
        bot.reply_to(message, f"The file size is a large, please send a smaller one: {str(e)}")


if __name__ == "__main__":
    bot.polling()
