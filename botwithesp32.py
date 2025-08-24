import cv2
import discord
from discord.ext import commands
import os
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import time
import requests
import numpy as np
from skimage.metrics import structural_similarity

load_dotenv()

bot_token = os.environ['DISCORD_BOT_TOKEN']
CHANNEL_ID = int(os.environ['CHANNEL_ID'])

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

known_faces_dir = "known_faces"

# Load known faces
known_faces = {}
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        known_faces[name] = cv2.imread(os.path.join(known_faces_dir, filename))


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera_url = "http://********<ip>/cam-lo.jpg"


def get_frame_from_esp32_cam(url):
    try:
        response = requests.get(url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)
        return True, frame
    except Exception as e:
        print(f"Error fetching frame: {e}")
        return False, None


async def detect_motion(channel):
    motion_detected = False
    is_start_done = False
    motion_frame_sent = False

    print("Waiting for 2 seconds...")
    time.sleep(2)
    frame_available, frm1 = get_frame_from_esp32_cam(camera_url)
    if not frame_available:
        print("Error fetching frame from ESP32-CAM.")
        return

    if len(frm1.shape) == 3:  # Check if the image is not grayscale
        frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

    while True:
        frame_available, frm2c = get_frame_from_esp32_cam(camera_url)
        if not frame_available:
            print("Error fetching frame from ESP32-CAM.")
            continue

        frm2 = cv2.cvtColor(frm2c, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            frm2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            for name, known_face in known_faces.items():
                known_face_gray = cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(
                    frm2[y:y+h, x:x+w], known_face_gray, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                loc = np.where(result >= threshold)
                if len(loc[0]) > 0:
                    await channel.send(f"Face detected: {name}")
                    break

        diff = cv2.absdiff(frm1, frm2)

        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        contors = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        contors = [c for c in contors if cv2.contourArea(c) > 25]

        if len(contors) > 5:
            cv2.putText(frm2c, "motion detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            if not motion_frame_sent:
                motion_frame_path = f'motion_detected_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
                await channel.send(f"Motion detected at {datetime.now().strftime('%Y-%m-%d_%H-%M-%S').split('_')[0]} at {datetime.now().strftime('%Y-%m-%d_%H-%M-%S').split('_')[1].replace('-',':')}")
                cv2.imwrite(motion_frame_path, frm2c)
                await channel.send(file=discord.File(motion_frame_path))
                os.remove(motion_frame_path)
                motion_frame_sent = True
            motion_detected = True
            is_start_done = False

        elif motion_detected and len(contors) < 3:
            if (is_start_done) == False:
                start = time.time()
                is_start_done = True
                end = time.time()

            end = time.time()

            print(end-start)
            if (end - start) > 4:
                frame_available, frame2 = get_frame_from_esp32_cam(camera_url)
                if not frame_available:
                    print("Error fetching frame from ESP32-CAM.")
                    continue
                spot_diff_result_path = spot_diff(frm1, frame2)
                await channel.send(file=discord.File(spot_diff_result_path))
                os.remove(spot_diff_result_path)
                if spot_diff_result_path == 0:
                    await channel.send("Nothing stolen.")
                else:
                    await channel.send("Theft/Missing objects detected!")

                return

        else:
            cv2.putText(frm2c, "no motion detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("winname", frm2c)

        _, frm1 = get_frame_from_esp32_cam(camera_url)
        if not frame_available:
            print("Error fetching frame from ESP32-CAM.")
            continue
        frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

    return


def spot_diff(frame1, frame2):

    frame1 = frame1[1]
    frame2 = frame2[1]

    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    g1 = cv2.blur(g1, (2, 2))
    g2 = cv2.blur(g2, (2, 2))

    (score, diff) = structural_similarity(g2, g1, full=True)

    print("Image similarity", score)

    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)[1]

    contors = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contors = [c for c in contors if cv2.contourArea(c) > 50]

    if len(contors):
        for c in contors:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    else:
        print("nothing stolen")
        return 0

    cv2.imshow("win1", frame1)
    spot_diff_result_path = f'spot_diff_result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
    cv2.imwrite(spot_diff_result_path, frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return spot_diff_result_path


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')


@bot.command()
async def start_surveillance(ctx):
    channel = ctx.channel
    await channel.send('Starting surveillance...')
    await detect_motion(channel)


@bot.command()
async def detect_faces(ctx):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        frame_available, frame = get_frame_from_esp32_cam(camera_url)
        if not frame_available:
            await ctx.send("Error fetching frame from ESP32-CAM.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            for name, known_face in known_faces.items():
                known_face_gray = cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(
                    gray[y:y+h, x:x+w], known_face_gray, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                loc = np.where(result >= threshold)
                if len(loc[0]) > 0:
                    await ctx.send(f"Face detected: {name}")
                    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    frame_path = f'frame_{now}.jpg'
                    cv2.imwrite(frame_path, frame)
                    await ctx.send(file=discord.File(frame_path))
                    os.remove(frame_path)
                    break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


@bot.command()
async def stop_detection(ctx):
    cv2.destroyAllWindows()

bot.run(bot_token)
