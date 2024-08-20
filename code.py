# Import các thư viện cần thiết
import cv2
import numpy as np
## Các thư viện để tạo GUI
from PIL import ImageTk
import PIL.Image
import math
from tkinter import *

# Define các biến cần thiết
#Lower_HSV / Upper_HSV / Name / thứ tự / giá trị BGR
#Lower/Upper là phạm vi giá trị màu trong không gian màu HSV
#OpenCV uses H: 0-179, S: 0-255, V: 0-255
COLOUR_BOUNDS = [
    [(0, 0, 0), (179, 255, 100), "BLACK", 0, (0, 0, 0)], # trong openCV Hue có giá trị từ 0-179
    [(0, 100, 20), (20, 255, 200), "BROWN", 1, (0, 51, 102)],
    [(0, 175, 20), (5, 255, 255), "RED", 2, (0, 0, 255)], 
    [(5, 150, 150), (15, 235, 255), "ORANGE", 3, (0, 128, 255)],  
    [(20, 70, 200), (70, 255, 255), "YELLOW", 4, (0, 255, 255)], 
    [(40, 100, 20), (70, 255, 255), "GREEN", 5, (0, 255, 0)],  
    [(90, 80, 2), (130, 255, 255), "BLUE", 6, (255, 0, 0)],  
    [(120, 100, 100), (150, 255, 255), "VIOLET", 7, (255, 0, 127)], 
    [(0, 0, 50), (175, 50, 255), "GRAY", 8, (128, 128, 128)], 
    [(0, 0, 168), (174, 111, 255), "WHITE", 9, (255, 255, 255)], 
]
# Màu đỏ có 2 phạm vi giá trị màu
RED_TOP_LOWER = (169, 175, 20)
RED_TOP_UPPER = (179, 255, 20)

# Các hàm trong chương trình
######################################## Dùng videocapture từ điện thoại
# Khởi tạo Haar Cascade và VideoCapture
def init():
    #kết nối với camera điện thoại
    cap = cv2.VideoCapture(0)
    url = "http://192.168.1.6:8080/video" # Địa chỉ IP của điện thoại
    cap.open(url)
    # Sử dụng haar cascade
    rectCascade = cv2.CascadeClassifier("haarcascade_resistors_0.xml") # Khởi tạo haar cascade để phân loại điện trở
    return (cap,rectCascade)
#Sử dụng haar cascade để nhận diện điện trở trong ảnh
def findResistors(img, rectCascade):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Chuyển sang ảnh mức xám để xử lý
    resDetected = []

    #Tìm kiếm các điện trở có trong khung hình
    ressFind = rectCascade.detectMultiScale(g_img,1.1,25)
    for (x,y,w,h) in ressFind: # (x,y,w,h) lưu tọa độ và kích thước của các điện trở
                               # được phát hiện bởi detectMultiScale
    # x,y đại diện cho góc trên cùng bên trái của hình chữ nhật giới hạn đối tượng
    # w,h là chiều rộng và chiều cao của hình chữ nhật    
        roi_gray = g_img[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]

        #dùng haar cascade detect lại lần nữa để lọc các kết quả chưa đúng
        secondCheck = rectCascade.detectMultiScale(roi_gray,1.01,5)

        if (len(secondCheck) != 0): # sau 2 lần kiểm tra đều cho kết quả phù hợp
            resDetected.append((np.copy(roi_color),(x,y,w,h)))
            # Thêm bản sao của roi_color và giá trị x,y,w,h tương ứng vào vào resDetected
    return resDetected
# Trả về giá trị True nếu contour hợp lệ và ngược lại
def validContour(cnt):
    #tìm contour có diện tích đủ lớn và đúng vs tỉ lệ kích thước
    if(cv2.contourArea(cnt) < 100):   # Loại bỏ những contour có diện tích quá nhỏ
        return False
    else:
        # tạo 1 hình chữ nhật bao quanh bằng boundingRect 
        x,y,w,h = cv2.boundingRect(cnt)   # gán x,y,w,h tương ứng vs tọa độ góc trái bên trên và chiều rộng, chiều cao hình chữ nhật
        aspectRatio = float(w)/h   
        # Xét tỉ lệ giữa chiều rộng và chiều cao xem có thỏa đk hay ko
        if (aspectRatio > 0.4):
            return False
    return True
#TÌm vạch màu trên điện trở
def findBands(resistorInfo):
    #resize ảnh điện trở thu dc
    resImg = cv2.resize(resistorInfo[0], (400, 200))                                       
    biFil_resImg = cv2.bilateralFilter(resImg,5,80,80) #Làm mượt ảnh bằng bilateral Filter để lọc nhiễu
    #Chuyển sáng HSV để xác định màu của vạch
    hsv_Img = cv2.cvtColor(biFil_resImg, cv2.COLOR_BGR2HSV) ###########################################
   
    # Nhị phân hóa ảnh bằng cách đặt ngưỡng
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(biFil_resImg, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 59,
                                  2)
    # Phương pháp phân ngưỡng ở trên không phù hợp cho nhiều trường hợp, 
    # như là ánh sáng không đồng đều trên ảnh. Trong trường hợp đó chúng ta dùng hàm adaptiveThreshold().
    # Phương thức này tính giá trị trung bình của các n điểm xung quanh pixel đó rồi trừ cho C chứ không lấy ngưỡng cố định 
    # (n thường là số lẻ, còn C là số nguyên bất kỳ).
    # cv2.ADAPTIVE_THRESH_MEAN_C ngưỡng bằng giá trị trung bình các điểm ảnh xung quanh điểm đang xét
    # cv2.THRESH_BINARY giá trị điểm ảnh vượt qua ngưỡng thì cho bằng 255, ngược lại là 0
    # 59 là kích thước ma trận
    # 2 là sai số
    thresh = cv2.bitwise_not(thresh) # CHuyển giá trị màu từ 0 sang 255 và ngược lại
    bandsPos = []
    checkColours = COLOUR_BOUNDS
    for clr in checkColours:
        mask = cv2.inRange(hsv_Img, clr[0], clr[1]) # Lọc màu với giá trị HSV đã có, nếu đúng thì chuyển sang màu trắng
        if (clr[2] == "RED"): # Gộp 2 phạm vi màu đỏ trong HSV
            redMask2 = cv2.inRange(hsv_Img, RED_TOP_LOWER, RED_TOP_UPPER) 
            mask = cv2.bitwise_or(redMask2,mask)# Gộp 2 mask
            #mask = mask + red_mask
        mask = cv2.bitwise_and(mask,mask,mask= thresh)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        # Tìm và trả về contour có trong ảnh

        #Lọc những Contour hợp lệ
        for k in range(len(contours)-1,-1,-1):
            if (validContour(contours[k])):
                # Tìm tọa độ 1 điểm trên contour 'k' và lưu tọa độ vào leftmost
                leftmost = tuple(contours[k][contours[k][:,:,0].argmin()][0])  
                bandsPos += [leftmost + tuple(clr[2:])]
                # bandsPos lưu thông tin về vị trí contour và các thông tin màu
            else: 
                # Loại bỏ contour không hợp lệ ra khỏi tuple
                # bằng cách chuyển sang list và xóa phần tử ko phù hợp
                # và chuyển lại sang tuple
                new_contours = list(contours) 
                new_contours.pop(k)
                contours = tuple(new_contours) 
        
        cv2.drawContours(biFil_resImg, contours, -1, clr[4], 3)  # Vẽ những contour tìm được lên màn hình
                                        
    cv2.imshow('Contour Display', biFil_resImg) 
    # Trả về 1 chuỗi contour hợp lệ đã được sắp xếp đúng thứ tự các màu từ trái sang phải
    return sorted(bandsPos, key=lambda tup: tup[0])
#Tính toán giá trị điện trở dựa trên nhũng vạch màu đẫ phát hiện được
def printResult(sortedBands, liveimg, resPos):
    x,y,w,h = resPos
    strVal = ""
    if (len(sortedBands) in [3,4,5]): # Nếu số vạch màu tìm được là 3 4 5 thì thực hiện code 
        for band in sortedBands[:-1]:
            strVal += str(band[3]) # TÍnh toán giá trị điện trở
        intVal = int(strVal)
        intVal *= 10**sortedBands[-1][3] # giữ lại vạch màu đơn vị để tích toán đơn vị
        #vẽ khung màu xanh nếu là điện trở có thể đọc đc vạch màu
        cv2.rectangle(liveimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(liveimg,str(intVal) + " OHMS",(x+w+10,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        return
    #vẽ khung màu đỏ nếu là điện trở ko thể đọc đc vạch màu
    cv2.rectangle(liveimg,(x,y),(x+w,y+h),(0,0,255),2)
######################################### Dùng ảnh có sẵn
def displayResults(sortedbands,liveimg):
    strvalue = ""
    if len(sortedbands) in [3, 4, 5]: # Nếu số vạch màu tìm được là 3 4 5 thì thực hiện code 
        for band in sortedbands[:-1]:
            strvalue += str(band[3])  # Tính toán giá trị điện trở
            #print("1",strvalue)
        intvalue = int(strvalue)
        #print("2",strvalue)
        intvalue *= 10 ** sortedbands[-1][3]  # giữ lại vạch màu đơn vị để tích toán đơn vị
        #print("3",sortedbands[-1][3])
        #print("2",intvalue)
        # In ra màn hình kết quả tính toán được
        cv2.putText(liveimg,str(intvalue) + " OHMS",(100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.imshow("Result",liveimg)
        return
    # In ra màn hình dòng chữ "Không xác định được" khi số lượng vạch màu đọc được không đủ yêu cầu
    cv2.putText(liveimg,"KHONG XAC DINH DUOC",(100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow("Result",liveimg)
    
def findBands1(img):
    img1 = cv2.bilateralFilter(img, 40, 90, 90)  # Ảnh được làm mượt bằng bilateralFilter để lọc nhiễu
    # là một bộ lọc hiệu quả cao trong việc loạt bỏ nhiễu mà vẫn giữ lại được các đường viền (cạnh) trong ảnh.
    cv2.imshow('bilateralFilter',img1)
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang mức ảnh xám để phân ngưỡng ảnh
    
    img_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # Chuyển đổi ảnh sang kênh màu HSV
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 79,2)  
    # Phương pháp phân ngưỡng ở trên không phù hợp cho nhiều trường hợp, 
    # như là ánh sáng không đồng đều trên ảnh. Trong trường hợp đó chúng ta dùng hàm adaptiveThreshold().
    # Phương thức này tính giá trị trung bình của các n điểm xung quanh pixel đó rồi trừ cho C chứ không lấy ngưỡng cố định 
    # (n thường là số lẻ, còn C là số nguyên bất kỳ).
    # cv2.ADAPTIVE_THRESH_MEAN_C ngưỡng bằng giá trị trung bình các điểm ảnh xung quanh điểm đang xét
    # cv2.THRESH_BINARY giá trị điểm ảnh vượt qua ngưỡng thì cho bằng 255, ngược lại là 0
    # 59 là kích thước ma trận
    # 2 là sai số
    thresh = cv2.bitwise_not(thresh) # CHuyển giá trị màu từ 0 sang 255 và ngược lại
    cv2.imshow('thresh',thresh)
    bandpos = []
    # Phát hiện màu sắc 
    for clr in COLOUR_BOUNDS:  
        mask = cv2.inRange(img_hsv, clr[0], clr[1]) # Lọc màu với giá trị HSV đã có, nếu đúng thì chuyển sang màu trắng
        if clr[2] == 'RED':  # Gộp 2 phạm vi màu đỏ trong HSV
            red_mask = cv2.inRange(img_hsv, RED_TOP_LOWER, RED_TOP_UPPER)
            mask = cv2.bitwise_or(red_mask, mask, mask) # Gộp 2 phạm vi
            #mask = mask + red_mask
        cv2.imshow('mask1',mask)
        mask = cv2.bitwise_and(mask,mask,mask= thresh)
        cv2.imshow('mask2',mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
        # Tìm và trả về contour có trong ảnh
        # Kết quả trả về là một danh sách Python của tất cả các đường viền trong hình ảnh. 
        # Mỗi đường viền riêng lẻ là một mảng Numpy gồm các tọa độ (x, y) của các điểm biên của đối tượng.
        #Lọc những Contour hợp lệ
        for i in range(len(contours) - 1, -1, -1):
            if validContour(contours[i]):
                # Tìm tọa độ 1 điểm trên contour 'k' và lưu tọa độ vào leftmost
                leftmost = tuple(
                    contours[i][contours[i][:, :, 0].argmin()][0])  
                bandpos += [leftmost + tuple(clr[2:])]
                # bandsPos lưu thông tin về vị trí contour và các thông tin màu
            else:
                # Loại bỏ contour không hợp lệ ra khỏi tuple
                # bằng cách chuyển sang list và xóa phần tử ko phù hợp
                # và chuyển lại sang tuple
                # new_contours = list(contours)
                # new_contours.pop(i)
                # contours = tuple(new_contours)
                contours = contours[i:i+1] #loai contour thu i khoi tuple
        cv2.drawContours(img1, contours, -1, clr[4], 3)  # Vẽ các contour tìm được
        
    cv2.imshow('Contour Display', img1)
    
    return sorted(bandpos,
                  key=lambda tup: tup[0])  # Trả về danh sách các contour đã được sắp xếp thứ tự
    
         
# GUI ###############
def mainPg():
    cap,rectCascade = init() # Mở Videocapture và khởi tạo haar cascade
    while(not (cv2.waitKey(1) == ord('q'))): # Bấm 'q' để thoát khỏi video
        # Lấy, mã hóa và trả về khung hình trong video 
        ret, img = cap.read() 
        resDetected = findResistors(img, rectCascade) 
        # Qua findResistors ta có 1 mảng chứa thông tin các điện trở detect được       
        for i in range(len(resDetected)):
            sortedBands = findBands(resDetected[i])
            # Từ mỗi điện trở tìm được ta có được các vạch màu theo thứ tự từ trái sang
            printResult(sortedBands, img, resDetected[i][1])
            # TÍnh toán và hiển thị giá trị của điện trở lên màn hình
        cv2.imshow("Frame",img)
    cap.release()
    cv2.destroyAllWindows()    
def closeGUI():
    root.destroy()
def mainImg():
    while(not (cv2.waitKey(1) == ord('q'))): # Bấm 'q' để thoát khỏi video
        image = cv2.imread('res56k.jpg') # Chọn ảnh đầu vào (res220/res1k/res56k/res1m)
        sortedbands = findBands1(image) # Tình vạch màu của điện trở
        # print(sortedbands)
        displayResults(sortedbands,image) # TÍnh toán hiển thị kết quả
    
    cv2.destroyAllWindows()
        
root=Tk()
root.title("FINAL PROJECT")
root.geometry("1196x677")
load = PIL.Image.open('03.png') #Chèn ảnh
render=ImageTk.PhotoImage(load)
img=Label(root,image=render)
img.place(x=0,y=0)
b=Button(root,text="VIDEO",font=('Poppins bold', 16),command=mainPg,pady=10)
b.place(x=1025,y=490)
c=Button(root,text="CLOSE",font=('Poppins bold', 16),command=closeGUI,pady=10)
c.place(x=1025,y=560)
a=Button(root,text="IMG",font=('Poppins bold', 16),command=mainImg,pady=10)
a.place(x=1120,y=490)
root.mainloop()
