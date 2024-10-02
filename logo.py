import cv2
import numpy as np

def detectLogo(template, img):
    
    templateg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # SURF extraction
    hessian_threshold = 30
    surf = cv2.SURF(hessian_threshold)
    kp, desc = surf.detect(imgg, None, useProvidedKeypoints = False)
    
    # KNN
    samples = np.array(desc)
    responses = np.arange(len(kp), dtype = np.float32)
    knn = cv2.KNearest()
    knn.train(samples, responses)
    
    # Loading template and searching for similar kp
    kp2, desc2 = surf.detect(templateg, None, useProvidedKeypoints = False)
    
    matched = 0
    total = 0
    for h,des in enumerate(desc2):
        des = np.array(des,np.float32).reshape((1,128))
        retval, results, neigh_resp, dists = knn.find_nearest(des,1)
        res,dist =  int(results[0][0]),dists[0][0]
        total += 1
        if dist<0.1: # draw matched keypoints in red color
            color = (0,0,255)
            matched += 1
        else:
            color = (255,0,0)
        #Draw matched key points on original image
        x,y = kp[res].pt
        center = (int(x),int(y))
        cv2.circle(img,center,2,color,-1)
        
        #Draw matched key points on template image
        x,y = kp2[h].pt
        center = (int(x),int(y))
        cv2.circle(template,center,2,color,-1)

    cv2.imwrite("../resources/template.jpg", template)
    cv2.imwrite("../resources/image.jpg", img)
    return matched / float(total)
    
template = cv2.imread("../resources/pictures/appleLogo.jpg")
img = cv2.imread("../resources/pictures/pic2.jpg")
print detectLogo(template, img)