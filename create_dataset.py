import os
import cv2
import torch

from argparse import ArgumentParser

coordinates = []
should_quit = False


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global coordinates
        coordinates.append((x, y))


def annotate_image(img, model, save_path, th=0.5):
    global coordinates
    global should_quit
    coordinates = []
    image_save_path = save_path + ".jpg"
    label_save_path = save_path + ".txt"
    img_w, img_h = img.shape[1], img.shape[0]

    # Save the image
    cv2.imwrite(image_save_path, img)

    results = model(img)

    df = results.pandas().xyxy[0]

    df = df[df["name"] == "person"]

    for _, row in df.iterrows():
        cv2.rectangle(
            img,
            (int(row["xmin"]), int(row["ymin"])),
            (int(row["xmax"]), int(row["ymax"])),
            (255, 155, 0),
            2,
        )

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    # Display the image
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create the label
    for coord in coordinates:
        print(coord)
        for idx, row in df.iterrows():
            if (
                coord[0] >= row["xmin"]
                and coord[0] <= row["xmax"]
                and coord[1] >= row["ymin"]
                and coord[1] <= row["ymax"]
            ):
                df.loc[idx, "class"] = 1
                df.loc[idx, "name"] = "referee"

    with open(label_save_path, "w") as f:
        for _, row in df.iterrows():
            f.write(
                f"{row['class']} {row['xmin']/img_w} {row['ymin']/img_h} {(row['xmax'] - row['xmin'])/img_w} {(row['ymax'] - row['ymin'])/img_h}\n"
            )

    coordinates = []
    return


def create_dataset(
    dir,
    save_dir,
    frame_interval=30,
):
    """
    Helps annotating images from a directory containing videos

    Parameters:"""

    # Set-up
    model = torch.hub.load(
        "2_referee-detection_TH/yolov7",
        "custom",
        "2_referee-detection_TH/yolov7/yolov7.pt",
        force_reload=True,
        source="local",
        trust_repo=True,
        verbose=False,
    )

    print("Model loaded")

    videos = os.listdir(dir)
    videos = [x for x in videos if "mp4" in x]

    os.makedirs(save_dir, exist_ok=True)
    # counter = len(os.listdir(save_dir)) + 1

    while True:
        video_name = videos.pop()
        video_path = os.path.join(dir, video_name)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        cnt = 0
        counter = len([x for x in os.listdir(save_dir) if video_name in x]) + 1
        while True:
            frame_exists, frame = video.read()
            if not frame_exists:
                break

            if cnt % frame_interval == 0:
                annotate_image(
                    frame,
                    model,
                    os.path.join(save_dir, video_name + "_" + str(counter)),
                )
                counter += 1

            cnt += 1

            if cnt == total_frames:
                break

    return


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dir",
        help="dir that cointains videos to annotate",
        type=str,
        default="/Users/tim/CS_master/Data_Science_Project/dataset/example_video_snippets",
    )
    parser.add_argument(
        "--save_dir",
        help="dir to save the images and labels",
        type=str,
        default="/Users/tim/CS_master/Data_Science_Project/dataset/referee-dataset",
    )
    args = parser.parse_args()

    create_dataset(args.dir, args.save_dir)
