import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import argparse
import cv2


def get_camera():
    return cv2.VideoCapture(0)

def get_neural_model():
    from keras.models import load_model
    print('Loading neural network model...')
    model = load_model('mask.h5')
    print('OK')
    return model

def get_logreg_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('~/Desktop/datasets/Skin_NonSkin.txt', delimiter='\t', names=['R','G','B','is_skin'])
    X = df[['R','G','B']]
    Y = df['is_skin']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print('Generating logistic regression model... ')
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                              multi_class='multinomial').fit(X_train, Y_train)
    print('OK')

    return clf

def neural_predict(model, flat_img):
    y = model.predict(flat_img)
    mask = np.array([np.argmax(y[i]) for i in range(len(y))])
    return mask

def run(method='logreg', show='skin'):

    if method == 'neural':
        model = get_neural_model()
    elif method == 'logreg':
        clf = get_logreg_model()
    else:
        raise Exception("method: use either 'logreg' or 'neural'.")

    cam = get_camera()

    while True:
        _, img = cam.read()
        h, w = img.shape[0], img.shape[1]
        reshaped = img.reshape(h*w, 3)
        
        # fast
        if method == 'logreg': 
            mask = clf.predict(reshaped) - 1

        # slow
        elif method == 'neural':
            mask = neural_predict(model, reshaped)

        else:
            raise Exception("method: use either 'logreg' or 'neural'.")

        mask = mask.reshape(h, w).astype(np.uint8)
        mask = 1 - mask
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get skin-colored pixels
        fin = cv2.bitwise_and(img, img, mask=mask)

        if show == 'skin':
            fin = cv2.cvtColor(fin, cv2.COLOR_RGB2BGR)
            cv2.imshow('color', fin)

        elif show == 'color':
            # Sort them into an array
            nonzero = np.where(fin!=0)
            nonzero_coordinates = nonzero[0][::3], nonzero[1][::3]
            color_samples = np.array(fin[nonzero_coordinates])
            mean = np.mean(color_samples, axis=0).astype('uint8')
            color = np.zeros((300, 300, 3), np.uint8)
            color[:] = mean
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imshow('color', color)

        else:
            print('Please set argument show to either "skin" or "color"')
        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

def show_kmeans():
    from sklearn.cluster import KMeans
    from mpl_toolkits.mplot3d import Axes3D

    cluster_num = 2
    kmeans = KMeans(n_clusters=cluster_num, init='k-means++').fit(nonzero_coordinates)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    points = np.c_[nonzero_coordinates, labels]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(nonzero_coordinates[:,0], nonzero_coordinates[:,1], nonzero_coordinates[:,2], c=points[:,3])
    ax.scatter(centroids)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('method', type=str,
                        help="method: choose either 'logreg' (fast) or 'neural' (terrible)")

    parser.add_argument('show', type=str,
                        help="show: choose either 'skin' or 'color'")

    args = parser.parse_args()
    run(method=args.method, show=args.show)

if __name__ == '__main__':
    main()