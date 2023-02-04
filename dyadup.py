import numpy as np
import cupy as cp

def dyadup1(x,*args):
    [b,c,v]=x.shape
    out=[]
    for i in range(b):
        out.append(dyadup(x[i],*args))
    out1=np.array(out)
    return out1
def dyadup(x,*args):
    """
    dyadic upsampling

    Calling Sequence
    ----------------
    Y=dyadup(x,[EVEN_ODD])
    Y=dyadup(M,[EVEN_ODD],[type])
    Y=dyadup(M,[type],[EVEN_ODD])

    Parameters
    ----------
    x : double vector
    M : double matrix
    EVEN_ODD : even or odd integer
    type : upsampling manner, 'r' for row, 'c' for column, and 'm' for row and column simutaneously.
    Y : upsampling result

    Description
    -----------
    dyadup is an utility function for dyadic upsampling. if EVEN_ODD is even, zeors will be put between input entries and output length will be two times input length minus one. Otherwise, additional two zeros will be put at the head and tail of output so the output length will be two times input length plus one. Default is odd. Optional argumet type is especially for matrix input upsampling.

    Examples
    --------
    a=rand(1,100)
    Y=dyadup(a)
    b=rand(25,25)
    Y=dyadup(b,'r',0)
    """
    m_orig = 0
    n_orig = 0
    if (np.size(x.shape) == 2 and np.min(x.shape) == 1):
        (m_orig,n_orig) = x.shape
        x = x.flatten()
    if (len(args) == 0 and np.size(x.shape) == 1):
        m1 = 1
        n1 = x.shape[0]
        m2 = 1
        n2 = n1 * 2 + 1
        output1 = np.zeros(n2*m2,dtype=np.float64)
        for i in range(len(x)):
            output1[1+(2*i)]=x[i]    
        if (m_orig > 1):
            output1.shape = (n2, 1)
        elif (n_orig > 1):
            output1.shape = (1, n2)
        return output1
    elif (len(args) == 1 and np.size(x.shape) == 1):
        # isinstance(args[0], int)
        m1 = 1
        n1 = x.shape[0]
        if ((args[0] % 2) == 0):
            m3 = 1
            n3 = n1 * 2 - 1
            output1 = np.zeros(n3*m3,dtype=np.float64)
            j=0
            for i in range(len(x)):
                output1[2*i]=x[j]
                j=j+1
            #_dyadup_1D_feed_odd(x, output1)
        else:
            m3 = 1
            n3 = n1 * 2 + 1
            output1 = np.zeros(n3*m3,dtype=np.float64)
            j=0
            for i in range(len(x)):
                output1[1+(2*i)]=x[j]
                j=j+1
            #_dyadup_1D_feed_even(x, output1)
        
        if (m_orig > 1):
            output1.shape = (n3, 1)
        elif (n_orig > 1):
            output1.shape = (1, n3)
        return output1
    elif (len(args) == 0 and np.size(x.shape) == 2):
        m1 = x.shape[0]
        n1 = x.shape[1]
        m2 = m1
        n2 = n1 * 2 + 1
        output1 = np.zeros((m2,n2),dtype=np.float64,order="F")
        for i in range(n1):
            output1[:,1+(2*i)]=x[:,i]
        #_dyadup_2D_feed_even_col(x.copy(order="F"), output1)
        return output1.copy(order="C")
    elif (len(args) == 1 and np.size(x.shape) == 2 and isinstance(args[0], str)):
        m1 = x.shape[0]
        n1 = x.shape[1]
        if (args[0] == "r"):
            m3 = m1 * 2 + 1
            n3 = n1
            output1 = np.zeros((m3,n3),dtype=np.float64,order="F")
            for i in range(m1):
                output1[1+(2*i),:]=x[i,:]
            #_dyadup_2D_feed_even_row(x.copy(order="F"), output1)
        elif (args[0] == "c"):
            m3 = m1
            n3 = n1 * 2 + 1
            output1 = np.zeros((m3,n3),dtype=np.float64,order="F")
            for i in range(n1):
                output1[:,1+(2*i)]=x[:,i]
            #_dyadup_2D_feed_even_col(x.copy(order="F"), output1)
        elif (args[0] == "m"):
            m3 = m1 * 2 + 1
            n3 = n1 * 2 + 1
            output1 = np.zeros((m3,n3),dtype=np.float64,order="F")
            for i in range(m1):
                for j in range(n1):
                    output1[1+(2*i),1+(2*i)]=x[i,j]
            #_dyadup_2D_feed_even(x.copy(order="F"), output1)
        else:
            raise Exception("Wrong input!!")
        return output1.copy(order="C")
    elif (len(args) == 1 and np.size(x.shape) == 2 and isinstance(args[0], int)):
        m1 = x.shape[0]
        n1 = x.shape[1]
        if ((args[0] % 2) == 0):
            m3 = m1
            n3 = n1 * 2 - 1
            output1 = np.zeros((m3,n3),dtype=np.float64,order="F")
            for i in range(n1):
                output1[:,2*i]=x[:,i]
            #_dyadup_2D_feed_odd_col(x.copy(order="F"), output1)
        else:
            m3 = m1
            n3 = n1 * 2 + 1
            output1 = np.zeros((m3,n3),dtype=np.float64,order="F")
            for i in range(n1):
                output1[:,1+(2*i)]=x[:,i]
            #_dyadup_2D_feed_even_col(x.copy(order="F"), output1)
        return output1.copy(order="C")
    elif (len(args) == 2 and np.size(x.shape) == 2):
        if (isinstance(args[0], int) and isinstance(args[1], str)):
            input_int = args[0]
            input_str = args[1]
        elif (isinstance(args[1], int) and isinstance(args[0], str)):
            input_int = args[1]
            input_str = args[0]
        else:
            raise Exception("Wrong input!!")
        m1 = x.shape[0]
        n1 = x.shape[1]
        if ((input_int % 2) == 0):
            if (input_str == "r"):
                m4 = m1 * 2 - 1
                n4 = n1
                output1 = np.zeros((m4,n4),dtype=np.float64,order="F")
                for i in range(m1):
                   output1[2*i,:]=x[i,:]
                
                #_dyadup_2D_feed_odd_row(x.copy(order="F"), output1)
            elif (input_str == "c"):
                m4 = m1
                n4 = n1 * 2 - 1
                output1 = np.zeros((m4,n4),dtype=np.float64,order="F")
                for i in range(n1):
                   output1[:,2*i]=x[:,i]
                #_dyadup_2D_feed_odd_col(x.copy(order="F"), output1)
            elif (input_str == "m"):
                m4 = m1 * 2 - 1
                n4 = n1 * 2 - 1
                output1 = np.zeros((m4,n4),dtype=np.float64,order="F")
                #_dyadup_2D_feed_odd(x.copy(order="F"), output1)
                for i in range(m1):
                    for k in range(n1):
                        output1[2*i,2*k]=x[i,k]
            else:
                raise Exception("Wrong input!!")
        else:
            if (input_str == "r"):
                m4 = m1 * 2 + 1
                n4 = n1
                output1 = np.zeros((m4,n4),dtype=np.float64,order="F")
                #_dyadup_2D_feed_even_row(x.copy(order="F"), output1)
            elif (input_str == "c"):
                m4 = m1
                n4 = n1 * 2 + 1
                output1 = np.zeros((m4,n4),dtype=np.float64,order="F")
                #_dyadup_2D_feed_even_col(x.copy(order="F"), output1)
            elif (input_str == "m"):
                m4 = m1 * 2 + 1
                n4 = n1 * 2 + 1
                output1 = np.zeros((m4,n4),dtype=np.float64,order="F")
                #_dyadup_2D_feed_even(x.copy(order="F"), output1)
            else:
                raise Exception("Wrong input!!")
        return output1.copy(order="C")
    else:
        raise Exception("Wrong input!!")
