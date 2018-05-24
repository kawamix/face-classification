import cv2
import re


def detect_and_save(input_file, output_file, cascade_file="haarcascade_frontalface_alt.xml"):
    """
    ある箇所(顔)を抽出して保存する
    :param input_file: 入力画像ファイルのパス
    :param output_file: 出力画像ファイルのパス
    :param cascade_file: カスケードファイルのパス
    :return:
    """
    trimmed_face_list = detect(input_file, cascade_file)
    for i, face in enumerate(trimmed_face_list):
        path = output_file if len(trimmed_face_list) == 1 else re.sub("\.(?=(jpg|png|gif))", "-" + str(i) + ".",
                                                                      output_file)
        cv2.imwrite(path, face)


def detect(input_file, cascade_file="haarcascade_frontalface_alt.xml", only_position=False):
    """
    顔を抽出する
    :param input_file: 入力画像ファイルのパス
    :param cascade_file: カスケードファイルのパス
    :param only_position: 位置情報だけ返すかどうか
    :return:
    """
    image_data = cv2.imread(input_file)
    cascade = cv2.CascadeClassifier(cascade_file)
    face_list = cascade.detectMultiScale(image_data, scaleFactor=1.1, minNeighbors=1, minSize=(32, 32))
    if only_position:
        return face_list
    trimmed_face_list = []
    for rect in face_list:
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        face = image_data[y:y + height, x:x + width]
        trimmed_face_list.append(face)
    return trimmed_face_list


if __name__ == '__main__':
    input_file = "./images/original.jpg"
    output_file = "./images/face_only.jpg"
    cascade_file = "haarcascade_frontalface_alt.xml"
    detect_and_save(input_file, output_file, cascade_file)
