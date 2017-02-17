import tensorflow as tf 
import numpy as np 
from collections import deque
import random
import os

theta = []    #weights and bias are stored here

###########################################################

def weight(shape,inp,outp):
	#Xavier initialization. To control the std-div of all layers
	rg = np.sqrt(6.0/(inp+outp))
	res = tf.random_uniform(shape,minval=-rg,maxval=rg,dtype=tf.float32)
	return tf.Variable(res)

def bias(shape):
	res = tf.constant(0.1,shape=shape)
	return tf.Variable(res)

###########################################################

def conv2D(x,size,inchn,outchn,stride=1,pad='SAME',activation=None):
	W = weight([size,size,inchn,outchn],size*size*inchn,outchn)
	b = bias([outchn])
	z = (tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=pad)+b)
	theta.append(W)
	theta.append(b)
	if activation==None:
		return z
	return activation(z)

def Fcnn(x,insize,outsize,activation=None):
	W = weight([insize,outsize],insize,outsize)
	b = bias([outsize])
	theta.append(W)
	theta.append(b)
	if activation==None:
		return tf.matmul(x,W)+b
	return activation(tf.matmul(x,W)+b)

###########################################################
def model(xs):
	#proposed model:
	#with 3x3 kernels (must be quite many) and stride=3
	conv1 = conv2D(xs,3,1,25,activation=tf.nn.relu)
	conv2 = conv2D(tf.concat(4,[xs,conv1]),3,26,52,activation=tf.nn.relu)
	flat1 = tf.reshape(tf.concat(4,[xs,conv1,conv2]),[-1,9*9*78])
	fc1 = Fcnn(flat1,9*9*78,300,activation=tf.nn.relu)
	fc2 = Fcnn(fc1,300,81)
	return fc2

def init_game():
	col = random.randint(0,9)
	row = random.randint(0,9)
	blank = np.zeros([9,9])
	blank[row,col] = 1
	return blank,[row,col]

def game_change_player(prev_state):
	return -1*prev_state

def checksmall(smallmap):
	sumrow = np.sum(smallmap,axis=0)
	sumcol = np.sum(smallmap,axis=1)
	sumcrs1 = smallmap[0][0]+smallmap[1][1]+smallmap[2][2]
	sumcrs2 = smallmap[0][2]+smallmap[1][1]+smallmap[2][0]
	if sumrow[0]==3 or sumrow[1]==3 or sumrow[2]==3 or sumcol[0]==3 or sumcol[1]==3 or sumcol[2]==3 or sumcrs1==3 or sumcrs2==3:
		return 1
	elif sumrow[0]==-3 or sumrow[1]==-3 or sumrow[2]==-3 or sumcol[0]==-3 or sumcol[1]==-3 or sumcol[2]==-3 or sumcrs1==-3 or sumcrs2==-3:
		return -1
	else:
		return 0

def checkoccupied(state):
	result = np.zeros([3,3])
	for i in range(3):
		for j in range(3):
			result[i][j] = checksmall(state[i*3:i*3+3,j*3:j*3+3])
	return result

def checkReward(prev_state,current_state):
	bigmap2 = checkoccupied(prev_state)
	bigmap = checkoccupied(current_state)
	sumrow = np.sum(bigmap,axis=0)
	sumcol = np.sum(bigmap,axis=1)
	sumcrs1 = bigmap[0][0]+bigmap[1][1]+bigmap[2][2]
	sumcrs2 = bigmap[0][2]+bigmap[1][1]+bigmap[2][0]
	if sumrow[0]==3 or sumrow[1]==3 or sumrow[2]==3 or sumcol[0]==3 or sumcol[1]==3 or sumcol[2]==3 or sumcrs1==3 or sumcrs2==3:
		return 5
	elif np.sum(bigmap)-np.sum(bigmap2) == 1:
		return 1
	else:
		return 0.1

def input_ex(s):
	try:
		result = input(s)
		result = int(result)
		return result
	except KeyboardInterrupt:
		raise
	except:
		return input_ex(s)

def printLine(line,reverselabel):
	labels = ['x','o']
	lbs = []
	for i in range(9):
		if line[i]*reverselabel==0.:
			lbs.append('.')
		if line[i]*reverselabel==1.:
			lbs.append(labels[0])
		if line[i]*reverselabel==-1.:
			lbs.append(labels[1])
	return lbs

def printState(state,reverselabel):
	for i in range(9):
		lbs = printLine(state[i],reverselabel)
		#print(lbs)
		print(lbs[0],lbs[1],lbs[2],' ',lbs[3],lbs[4],lbs[5],' ',lbs[6],lbs[7],lbs[8],' ')
		if i%3==2:
			print()

def next_state(statein,big,number):
	state = np.copy(statein)
	bigrow = int(big/3)
	bigcol = int(big%3)
	smallrow = number/3
	smallcol = number%3
	row = bigrow*3+smallrow
	col = bigcol*3+smallcol
	if state[row][col]!=0.:
		number = int(input_ex('Illegal number, reinput:'))-1
		return next_state(state,big,number)
	state[row][col] = 1.
	return state,[big,number]

def init_blank():
	os.system('cls')
	blank = np.zeros([9,9])
	printState(blank,-1)
	big = int(input_ex('Grid:'))-1
	number = int(input_ex('Number:'))-1
	state,prev_move = next_state(blank,big,number)
	return state,prev_move



def next_move(statein,prev_move):
	state = np.copy(statein)
	gridstate = checkoccupied(state)
	#print(gridstate)
	grid = prev_move[1]
	gridrow = int(grid/3)
	gridcol = int(grid%3)
	if gridstate[gridrow][gridcol]!=0.:
		s = '\nGrid '+str(grid+1)+' is occupied\nchoose another grid:'
		grid = int(input_ex(s))-1
		return next_move(state,[prev_move[0],grid])
	print('At grid',grid+1)
	number = int(input_ex('number:'))-1
	return next_state(state,grid,number)

#######################################
#play game
state,pre_move = init_blank()
rev = 1
labels = ['x','o']
while True:
	rew = 0
	state = game_change_player(state)
	rev = -rev
	os.system('cls')
	printState(state,rev)
	if rev == 1 :
		print('Term for',labels[0])
	else:
		print('Term for',labels[1])
	state2,pre_move = next_move(state,pre_move)
	rwd = checkReward(state,state2)
	# os.system('cls')
	# printState(state2,rev)
	# print('Move reward:',rwd)
	state = state2
	if rwd==5:
		if rev == 1 :
			print(labels[0],'win the game!')
		else:
			print(labels[1],'win the game!')
		input('Press any key to restart game...')
		state,pre_move = init_blank()
	# input()


OBSERVE = 10000
EXPLORE = 500000
EPSILON = 1
EPSILON2 = 0.001
MEMORY = 100000
def trainAI():
	#define train method
	xs = tf.placeholder(tf.float32,[None,9,9,1])
	action = tf.placeholder(tf.float32,[None,9])
	Qaction = tf.reduce_sum(tf.mul(action,model(xs)),reduce_indicies=1)
	y = tf.placeholder(tf.float32,[None])
	loss = tf.reduce_mean(tf.square(y-Qaction))
	train = tf.train.AdamOptimizer(0.0001).minimize(loss)

	#initialize
	D = deque()
	state,prev_move = init_game()
	epsi = EPSILON
	t = 0
	while True:
		if random.random() <=epsi:
			print('------Random move---------')
			actionInd = random.randrange(0,9)
