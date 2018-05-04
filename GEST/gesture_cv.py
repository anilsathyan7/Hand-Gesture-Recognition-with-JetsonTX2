import caffe
import numpy as np
import cv2
import Image
import os
#nvcamerasrc
#gst = "/dev/video1 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

#gst-launch-1.0 v4l2src device="/dev/video1" ! 'video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1' ! nvvidconv flip-method=0 ! 'video/x-raw, format=(string)I420' ! videoconvert ! 'video/x-raw, format=(string)BGR' ! fakesink

MODEL_FILE='/home/nvidia/GEST/deploy.prototxt'
WEIGHT_CAFFEMODEL='/home/nvidia/GEST/snapshot_iter_17255.caffemodel'
MEAN_FILE='/home/nvidia/GEST/mean.npy'
LABEL='/home/nvidia/GEST/post_label.txt'

caffe.set_mode_gpu()
caffe.set_device(0)

def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


def main():

	cap = cv2.VideoCapture(1)

	net = caffe.Net(MODEL_FILE,WEIGHT_CAFFEMODEL,caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load(MEAN_FILE).mean(1).mean(1))

	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
	transformer.set_raw_scale('data', 255.0)

	net.blobs['data'].reshape(1,3,224,224)
	count=1
	res='start'

	while(True):
   	 # Capture frame-by-frame
		count += 1
		ret, frame = cap.read()
		frame = binaryMask(frame)
		cv2.putText(frame,res,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
		cv2.imshow('frame',frame)
		
		if res=="one":
			os.system("bash 1.sh")
		elif res == "two":
			os.system("bash 2.sh")
		elif res == "three":
			os.system("bash 3.sh")
		elif res == "four":
			os.system("bash 4.sh")
		elif res == "five":
			os.system("bash 5.sh")
		elif res == "zero":
			os.system("bash 0.sh")
 

		if count % 5 == 0:        #process every 5th frame
			cv2.imwrite('temp.png', frame)
			img = caffe.io.load_image('temp.png') #/media/anilsathyan7/work/imdb/47/tomato-109.jpg
			net.blobs['data'].data[...] = transformer.preprocess('data', img)

			output = net.forward()

			output_prob = output['softmax'][0]
			print output['softmax'].argmax()


			label_mapping = np.loadtxt(LABEL, str, delimiter='\t')
			best_n = net.blobs['softmax'].data[0].flatten().argsort()[-1:-6:-1]

			top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
			res = label_mapping[top_inds][0][1]
			print "It looks like a ",label_mapping[top_inds][0][1]

		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
