import multiprocessing as mp
import os

# #多线程测试，不能用满4个核
# def loop():
#     x=0
#     while True:
#         x = x*1

# for i in range(100):
#     t = threading.Thread(target =loop)
#     t.start()

# def f(name):
#     print("hello",name)

# if __name__ =='__main__':
#     p = mp.Process(target = f,args =('bob',))
#     p.start()
#     p.join()

# def run_proc(name):
#     print("Run child process %s(%s)"%(name,os.getpid()))

# if __name__=='__main__':
#     print("Parent process %s"%os.getpid())
#     p = mp.Process(target =run_proc,args=('test',))
#     print("Child process begin")
#     p.start()
#     p.join()
#     print("OVER")

def 