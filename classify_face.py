import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import detect_face as df
import chainer
from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from logging import getLogger, StreamHandler, INFO, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class CNN(Chain):
    def __init__(self, output_size=2, dropout=0.2):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=3, out_channels=128, ksize=5)
            self.conv2 = L.Convolution2D(in_channels=128, out_channels=256, ksize=5)
            self.l1 = L.Linear(1024, 512)
            self.l2 = L.Linear(512, output_size)
        self.dropout_ratio = dropout

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 5)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 5)
        with chainer.using_config('train', train):
            h3 = F.dropout(F.relu(self.l1(h2)), ratio=self.dropout_ratio)
        return self.l2(h3)

    def predict(self, x):
        y = self(x, train=False)
        return y.data.argmax()


def process_face_img():
    """
    顔のトリミングと保存。
    ただし他の人の顔が混ざっている場合があるので、書き出されたデータに対して手動で選別したほうがよい
    :return:
    """
    for original_dir, face_dir in data_path:
        assert os.path.isdir(original_dir) and os.path.isdir(face_dir)
        for input_file in os.listdir(original_dir):
            input_file_path = os.path.join(original_dir, input_file)
            output_file_path = os.path.join(face_dir, input_file)
            df.detect_and_save(input_file_path, output_file_path)


def load_data():
    """
    画像データ読み込み
    :return:
    """
    img_data = []
    label_data = []
    max_size = max([len(os.listdir(data[1])) for data in data_path])
    for i, input_dir in enumerate(data_path):
        for j, input_file in enumerate(os.listdir(input_dir[1])):
            if j >= max_size - 1:
                break
            image = cv2.imread(os.path.join(input_dir[1], input_file))
            image = cv2.resize(image, (in_size, in_size), interpolation=cv2.INTER_CUBIC)
            image = image.transpose(2, 0, 1) / 255.  # 正規化
            img_data.append(image)  # 画像
            label_data.append(i)  # ラベル
    return img_data, label_data


def train():
    """
    モデルを学習する
    :return:
    """
    # データ処理
    img_data, label_data = load_data()
    img_data = np.array(img_data, dtype=np.float32)
    label_data = np.array(label_data, dtype=np.int32)
    # 訓練用特徴行列、テスト用特徴行列、訓練用正解ラベル、テスト用正解ラベル
    X_train, X_test, Y_train, Y_test = train_test_split(img_data, label_data, test_size=0.2)

    model = CNN(dropout=0.5)
    # 学習
    for i in range(epoch):
        logger.info("========== EPOCH {} ==========".format(i + 1))
        optimizer = optimizers.Adam()
        optimizer.setup(model)
        total_train_loss = 0
        total_train_accuracy = 0
        # 訓練
        for j in range(0, len(X_train), batch_size):
            x_data = X_train[j:j + batch_size]
            t_data = Y_train[j:j + batch_size]
            y_data = model(x_data)
            # 損失計算
            logger.debug("x_data:{}".format(x_data.data))
            logger.debug("t_data:{}".format(t_data.data))
            logger.debug("y_data:{}".format(y_data.data))
            loss = F.softmax_cross_entropy(y_data, t_data)
            acc = F.accuracy(y_data, t_data)
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            total_train_loss += loss.data * len(x_data)
            total_train_accuracy += acc.data * len(x_data)
        logger.info("train_loss:{}, train_accuracy:{}".format(total_train_loss / len(X_train),
                                                              total_train_accuracy / len(X_train)))

        total_test_loss = 0
        total_test_accuracy = 0
        # テスト
        for j in range(0, len(X_test), batch_size):
            x_data = X_test[j:j + batch_size]
            t_data = Y_test[j:j + batch_size]
            y_data = model(x_data, train=False)
            loss = F.softmax_cross_entropy(y_data, t_data)
            acc = F.accuracy(y_data, t_data)
            total_test_loss += loss.data * len(x_data)
            total_test_accuracy += acc.data * len(x_data)
        logger.info("test_loss:{}, test_accuracy:{}".format(total_test_loss / len(X_test),
                                                            total_test_accuracy / len(X_test)))
    chainer.serializers.save_npz(model_path, model)


def predict(input_file, model):
    """
    画像ファイルを入力して判別する
    :param input_file: 画像ファイル
    :param model: 学習済みモデル
    :return:
    """
    face_list = df.detect(input_file)
    face_position_list = df.detect(input_file=input_file, only_position=True)
    for orig_image, rect in zip(face_list, face_position_list):
        image = cv2.resize(orig_image, (in_size, in_size), interpolation=cv2.INTER_CUBIC)
        image = image.transpose(2, 0, 1) / 255.  # 正規化
        index = model.predict(np.array([image], dtype=np.float32))
        name = label_list[index]
        print("{}さんと予測しました。".format(name[0]))
        cv2.imshow('image', paint_on_face(input_file, rect, name[1]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def paint_on_face(input_file, rect, text=None):
    """
    四角形と名前を画像上に描画
    :param input_file: 画像
    :param rect: 顔の位置情報
    :param text: 描画する名前(テキスト)
    :return:
    """
    image = cv2.imread(input_file)
    image = cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 5)
    if text:
        image = cv2.putText(image, text, (rect[0], rect[1] + rect[3] + 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return image


# オリジナルと顔抽出画像の各ディレクトリパス
# process_face_img()で顔を抽出した画像に書き出す。ただし学習に使う前に人手による選別を推奨。
data_path = [("./images/fujita/fujita-original/", "./images/fujita/fujita-face/"),
             ("./images/kubota/kubota-original/", "./images/kubota/kubota-face/")]

label_list = [("藤田茜", "Akane Fujita"), ("久保田未夢", "Miyu Kubota")]  # ラベル
model_path = "model_fujikubo.npz"  # 保存するモデルのパス
in_size = 64  # 画像サイズ
epoch = 30
batch_size = 20
TRAIN = False

if __name__ == '__main__':
    if TRAIN:
        # train
        train()
    else:
        # predict
        model_ = CNN(dropout=0.5)
        chainer.serializers.load_npz(model_path, model_)
        while True:
            predict(input("INPUT PICTURE FILE PATH>>"), model_)
