# Indentify_Nogizaka46
VGG16をファインチューニングしたモデルで, 乃木坂46に関連するメンバー3人(白石さん,橋本さん,西野さん)を  
識別します.

## 用意するもの  
- `haarcascade_frontalface_alt.xml`  
  顔認識に利用します.  
  `./cv2/`ディレクトリに保存してください.  
- `オリジナルの画像`  
  メンバー3人のオリジナル画像.  
  `./image/original/`内に`メンバー名_original`ディレクトリを作成し保存してください.  
  
## 使用方法  
1. `face_cut.py`でオリジナル画像から顔部分のみ切り取ります.  
1. `devise_data.py`で学習データとテストデータに分けます.  
1. `inflated.py`を用いて学習データを水増し(元データ数の15倍)させます.  
1. `train.py`もしくは`train_finetuning.py`で学習を行い, 精度を出力させます.  
   `train_finetuning.py`はVGG16をファインチューニングしたモデルです.
1. `test.py`は`./image/test_data/original_test/`にある画像に対し予測を行うことができます.  

## 参考にしたページ  
https://aidemy.hatenablog.com/entry/2017/12/17/214715  
https://qiita.com/nirs_kd56/items/bc78bf2c3164a6da1ded  
