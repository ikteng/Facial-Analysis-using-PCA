from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    #load the file
    x = np.load(filename)

    #find mean
    mean = np.mean(x, axis=0)

    #find center
    center=x-mean

    return center

def get_covariance(dataset):
    #1024 x 1024 = d x d
    XTX=np.dot(np.transpose(dataset),dataset)
    convariance=XTX/(len(dataset)-1)

    return convariance
    
def get_eig(S, m):
   n=len(S)
    #get the largest eigval, eivectors (ascending order)
   eigval,eigvector=eigh(S,subset_by_index=[n-m,n-1])
   diagonal=np.diag(eigval)

   return np.flip(diagonal),np.fliplr(eigvector)

def get_eig_prop(S, prop):
    sum_lambda_j=0

    #get only eighval
    eighval=eigh(S,eigvals_only=True)

    #sum all the eighvals
    sum_lambda_j=np.sum(eighval)

    #proportion of variance by setting a min & max
    w,v=eigh(S,subset_by_value=[prop * sum_lambda_j,np.inf])

    diagonal=np.diag(w)

    return np.flip(diagonal),np.fliplr(v)

def project_image(image, U):
    #xi_pca = U * aij (from 5.4.2)
    a_ij=np.dot(np.transpose(U),image)
    xi_pca=np.dot(U,a_ij)

    return xi_pca
    
def display_image(orig, proj):
    #reshaping the images from 1024 x 1024 to 32 x 32
    #transpose to rotate clockwise
    reshape_o=np.transpose(np.reshape(orig,(32,32)))
    reshape_p=np.transpose(np.reshape(proj,(32,32)))

    #subplot
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    
    #set titles
    ax1.set_title("Original")
    ax2.set_title("Projection")

    #imshow
    original=ax1.imshow(reshape_o,aspect='equal')
    projection=ax2.imshow(reshape_p,aspect='equal')

    #color bar
    fig.colorbar(original,ax=ax1)
    fig.colorbar(projection,ax=ax2)

    return fig, ax1, ax2

#main method
if __name__=="__main__":
    center = load_and_center_dataset("YaleB_32x32.npy")

    convariance=get_covariance(center)
    
    # can change the number, the higher the closer to the original
    Lambda, U = get_eig(convariance,50)

    #Lambda,U = get_eig_prop(convariance,0.07)

    projections=[]
    #print(len(center))

    # get the first 5 images
    for i in range(5):
        projection = project_image(center[i], U)
        projection_str = str(projection)  # Convert the array to a string for comparison
        #print(projection)
        
        if projection_str not in projections:
            projections.append(projection_str)
            fig, ax1, ax2 = display_image(center[i], projection)
            plt.show()
            plt.close()
