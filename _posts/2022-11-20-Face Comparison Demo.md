---
layout:     post                    # ʹ�õĲ��֣�����Ҫ�ģ�
title: Face Comparison Demo                 # ���� 
subtitle:   һ�����׵������ȶ�ϵͳ #������
date:       2022-11-14              # ʱ��
author:     zichuana                     # ����
header-img: img/2022-11-20/page.jpg   #��ƪ���±��ⱳ��ͼƬ
catalog: true                       # �Ƿ�鵵
tags:                               #��ǩ
    - facenet
    - flask
    - demo
---
>����Facenet-pytorch��ܿ���ʵ�ּ�demo

��׼������ļ������˹����ܵ�ʱ��ע�⵽�в������»���`Facenet-pytorch`������ͼ����бȶԣ�ǳ����ʱ�������д���򵥵�demo(���ò�˵Facenet׼ȷ���涥)��  
![image](https://raw.githubusercontent.com/Zichuana/Face-Comparison-System/main/display1.png)
![image](https://raw.githubusercontent.com/Zichuana/Face-Comparison-System/main/display2.png)
����Դ���ϴ���github��  
[https://github.com/Zichuana/Face-Comparison-System](https://github.com/Zichuana/Face-Comparison-System)  
���õ������ɿ⣬�����Ӧ�汾��  
```
Flask                        2.2.1
opencv-python                4.6.0.66
torch                        1.12.1
facenet-pytorch              2.5.2
```
### ����ʵ��
����ʵ�ֱ�д��`app.py`�ڣ������չʾĿ¼�ṹ����������Ⲣ��ʼ����Ŀ�� 

```
from flask import Flask, jsonify, render_template, request
from datetime import timedelta
import cv2
import io
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__, static_url_path="/")

# �Զ�����ģ���ļ�
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
# ���þ�̬�ļ��������ʱ��
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
�����������������  
```
def load_known_faces(dstImgPath, mtcnn, resnet):
    aligned = []
    knownImg = cv2.imread(dstImgPath)  # ��ȡͼƬ
    face = mtcnn(knownImg)  # ʹ��mtcnn������������ء��������顿
    print(face)
    if face is not None:
        aligned.append(face[0])
    print(aligned)
    aligned = torch.stack(aligned).to(device)
    with torch.no_grad():
        known_faces_emb = resnet(aligned).detach().cpu()  # ʹ��resnetģ�ͻ�ȡ������Ӧ����������
    # print("\n������Ӧ����������Ϊ��\n", known_faces_emb)
    return known_faces_emb, knownImg
```
�������������������ŷ�Ͼ��룬������ֵ���ж��Ƿ�Ϊͬһ��������  
```
def match_faces(faces_emb, known_faces_emb, threshold):
    isExistDst = False
    distance = (known_faces_emb[0] - faces_emb[0]).norm().item()
    if (distance < threshold):
        isExistDst = True
    return distance, isExistDst
```
�����ҳ��Ϊindex.html(ʵ������ʾҲֻ��Ҫ��һ��ҳ��)��  
```
@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template("index.html")
```
����`/predict`·�ɣ��Դ�`request`��ȡ��������ͼƬ���з�������������ݸ�ǰ�ˡ�  
```
@app.route("/predict", methods=["GET", "POST"])
@torch.no_grad()
def predict():
    info = {}
    try:
        image1 = request.files["file0"]
        image2 = request.files["file1"]
        img_bytes1, img_bytes2 = image1.read(), image2.read()
        image1, image2 = Image.open(io.BytesIO(img_bytes1)), Image.open(io.BytesIO(img_bytes2))
        image_path1, image_path2 = './data/a.png', './data/b.png'
        image1.save(image_path1)
        image2.save(image_path2)
        mtcnn = MTCNN(min_face_size=12, thresholds=[0.2, 0.2, 0.3], keep_all=True, device=device)
        # InceptionResnetV1ģ�ͼ��ء����ڻ�ȡ��������������
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        MatchThreshold = 0.8  # ������������ƥ����ֵ����

        known_faces_emb, _ = load_known_faces(image_path1, mtcnn, resnet)  # ��֪����ͼ
        # bFaceThin.png  lyf2.jpg
        faces_emb, img = load_known_faces(image_path2, mtcnn, resnet)  # ���������ͼ
        distance, isExistDst = match_faces(faces_emb, known_faces_emb, MatchThreshold)  # ����ƥ��
        info["oushi"] = "����������ŷʽ����Ϊ��{}".format(distance)
        info["fazhi"] = "���õ�������������ƥ����ֵΪ��{}".format(MatchThreshold)
        print("OK")
        if isExistDst:
            boxes, prob, landmarks = mtcnn.detect(img, landmarks=True)  # ���������򣬸��ʣ�5�������ؼ���
            info["result"] = '����ŷ�Ͼ���С��ƥ����ֵ��ƥ�䣡���жϷ�ʽ����һ���ˣ�'
        else:
            info["result"] = '����ŷ�Ͼ������ƥ����ֵ����ƥ�䣡���жϷ�ʽ�²���һ���ˣ�'
    except Exception as e:
        info["err"] = str(e)
    return jsonify(info)  # json��ʽ����ǰ��
```
### ��ʾ����
�������һ��htmlģ�壬Ŀ¼�ṹ���±�����̬�ļ���`static`�ڵ����ݡ� 
![image](/img/2022-11-20/a.png)
����`data`�ļ����ݴ�ǰ�˻�ȡ����ͼƬ���Ա���ܽ��еĴ���  
`index.html`�ļ�����ֱ������ģ�壬��ģ��������������ݡ�  
ͨ��`input`��ȡͼ��������Ҫ������бȶԵ�ͼƬ����ʼ��չʾ����ķ�������Ĳ����Լ���ť�������`class`����ģ���д����ʵ�ֵ����ť����`test0()`����ͼ�񴫸���ˣ�����ȡ�����    
```
    <!-- Start Upcoming Events Section -->
    <section class="bg-upcoming-events">
        <div class="container">
            <div class="row">
                <div class="upcoming-events">
                    <div class="section-header">
                        <h2>&#129409;</h2>
                        <p>�ϴ���Ҫ�ж��Ƿ�Ϊͬһ�˵���������ͼ�񣬵�����԰�ť���бȶԷ���</p>
                    </div>
                    <!-- .section-header -->
                    <div class="row">
                        <div class="col-lg-6">
                            <div>
                                <!--                 href="javascript:;"-->
                                <input href="javascript:;" class="btn btn-default" tabindex="0" type="file" name="file"
                                       id="file0">

                                </input>
                                <p></p>
                                <img src="" id="img0">
                            </div>
                        </div>
                        <!-- .col-lg-6 -->
                        <div class="col-lg-6">
                            <div>
                                <!--                 href="javascript:;"-->
                                <input href="javascript:;" class="btn btn-default" tabindex="0" type="file" name="file"
                                       id="file1">
                                </input>
                                <p></p>
                                <img src="" id="img1">
                            </div>
                        </div>
                        <!-- .col-lg-6 -->
                        <p></p>
                        <div>
                            <!--                style="margin-top:20px;width: 35rem;height: 30rem; padding-left: 20px"-->
                            <input class="btn btn-default" type="button" id="b0"
                                   onclick="test0()" style="color: #000000"
                                   value="Ԥ��">
                            <p></p>
                            <pre id="out">&#129300;���Ԥ���ȡ���</pre>
                            <!--                <pre id="out" style="width:320px;height:50px;line-height: 50px;margin-top:20px;"></pre>-->
                        </div>
                    </div>
                    <!-- .row -->
                </div>
                <!-- .upcoming-events -->
            </div>
            <!-- .row -->
        </div>
        <!-- .container -->
    </section>
    <!-- End Upcoming Events Section -->s
```
��ĩβ����`<script>`Ԫ�أ���д`test0()`���ܣ���ͼƬ`img0`��`img1`���ݸ���ˡ�  
```
<script type="text/javascript">
    $("#file0").change(function () {
        var objUrl = getObjectURL(this.files[0]);//��ȡ�ļ���Ϣ
        console.log("objUrl = " + objUrl);
        if (objUrl) {
            $("#img0").attr("src", objUrl);
        }
    });
    $("#file1").change(function () {
        var objUrl = getObjectURL(this.files[0]);//��ȡ�ļ���Ϣ
        console.log("objUrl = " + objUrl);
        if (objUrl) {
            $("#img1").attr("src", objUrl);
        }
    });

    function test0() {
        var fileobj0 = $("#file0")[0].files[0];
        var fileobj1 = $("#file1")[0].files[0];
        var form = new FormData();
        form.append("file0", fileobj0);
        form.append("file1", fileobj1);
        var out = '';
        var fazhi = '';
        var oushi = '';
        $.ajax({
            type: 'POST',
            url: "predict",
            data: form,
            async: false,       //ͬ��ִ��
            processData: false, // ����jqueryҪ����data����
            contentType: false, //����jquery����Ҫ��������ͷ����contentType������
            success: function (arg) {
                console.log(arg)
                out = arg.fazhi + '\n' + arg.oushi + '\n' + arg.result
                err = arg.err
            }, error: function () {
                console.log("��̨�������");
            }
        });

        // if(oushi!==undefined){document.getElementById("oushi").innerText = oushi;}
        // if(fazhi!==undefined){document.getElementById("fazhi").innerText = fazhi;}
        if (err !== undefined) {
            document.getElementById("out").innerText = err;
        }
        if (out !== undefined) {
            document.getElementById("out").innerHTML = out;
        }

    }

    function getObjectURL(file) {
        var url = null;
        if (window.createObjectURL != undefined) {
            url = window.createObjectURL(file);
        } else if (window.URL != undefined) { // mozilla(firefox)
            url = window.URL.createObjectURL(file);
        } else if (window.webkitURL != undefined) { // webkit or chrome
            url = window.webkitURL.createObjectURL(file);
        }
        return url;
    }
</script>
```
