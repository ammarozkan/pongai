import tensorflow as tf
import numpy as np
import pygame

#Checkpoint System
checkpoint_path="pong/pong.ckpt"
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

#AI
    #model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5)),
    tf.keras.layers.Dense(5,activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
#load Checkpoint
model.load_weights(checkpoint_path)

inputs=[]
outputs=[]

def resArrays():
    inputs=[]
    outputs=[]
counter = 0

bs = 5
#Screen and drawning
size = width,height = 400,400
    #init
pygame.init()
screen = pygame.display.set_mode((int(width),int(height)))

    #DrawningColor
c_open= (   255,255,255)
c_off = (   0,  0,  0)

def drawRect(x,y,width,height):
    pygame.draw.rect(screen,c_open,pygame.Rect(int(x),int(y),int(width),int(height)))

#Object
class Object:
    def __init__(self,startX=45,startY=45,startSX=10,startSY=10,startSpeed = 1):
        self.pos,self.size = [startX,startY],[startSX,startSY]
        self.speed = startSpeed
    
    def getCol(self, cpX,cpY):
        return (self.pos[0]<cpX) and (self.pos[0]+self.size[0]>cpX) and (self.pos[1]<cpY) and (self.pos[1]+self.size[1]>cpY)
    
    def getColBlock(self,pos,size):
        return self.getCol(pos[0],pos[1]) or self.getCol(pos[0]+size[0],pos[1]) or self.getCol(pos[0],pos[1]+size[1]) or self.getCol(pos[0]+size[0],pos[1]+size[1])

    def print(self):
        drawRect(self.pos[0],self.pos[1],self.size[0],self.size[1])
    
    def move(self,mX,mY):
        self.pos[0]+=int(mX*self.speed)
        self.pos[1]+=int(mY*self.speed)


#Control
control1=[0.0,0.0]
control2=[0.0,0.0]
score=[0,0]

player1 = Object(0,40,10,100,6)
player2 = Object(390,40,10,100,6)
ball    = Object(200,200,10,10,2)
ballV   = [1.0,0.0]

def setVector(l):
    ballV[1]+=0.03*l

#System
clock = pygame.time.Clock()
isOpen = True
while isOpen:
    #Controls
    for event in pygame.event.get():
        if event.type == pygame.QUIT: isOpen=False
    
    keys=pygame.key.get_pressed()
    if(keys[pygame.K_w]):control1[1]=1.0
    else:control1[1]=0.0
    if(keys[pygame.K_s]):control1[0]=1.0
    else:control1[0]=0.0

    if(keys[pygame.K_o]):control2[1]=1.0
    else:control2[1]=0.0
    if(keys[pygame.K_l]):control2[0]=1.0
    else:control2[0]=0.0
    #control2 = [0.0,0.0]
    inputToAI=[ball.pos[0]/400,ball.pos[1]/400,ballV[0]/5,ballV[1]/5,player2.pos[1]/400]
    AIControls=model.predict([inputToAI])
    control2=AIControls[0]
    print("AI:",AIControls)

    #Train Values
    control=np.array(control1)
    values=np.array ([((400-ball.pos[0])/400),ball.pos[1]/400,((5-ballV[0])/5),ballV[1]/5,player1.pos[1]/400])
    inputs.append(values)
    outputs.append(control)
    
    #inputs=np.insert(inputs,values,axis=2)
    #outputs=np.instert(outputs,control,axis=2)
    
    #Move
    moveLimit = 0.3
    if(control1[0]>moveLimit):player1.move(0,1)
    if(control1[1]>moveLimit):player1.move(0,-1)

    if(control2[0]>moveLimit):player2.move(0,1)
    if(control2[1]>moveLimit):player2.move(0,-1)

    ball.move(ballV[0],ballV[1])
    
    #Collision
    if(ball.pos[1]<0) : ballV[1]*=-1
    elif(ball.pos[1]+ball.size[1]>400):ballV[1]*=-1

    if(player1.getColBlock(ball.pos, ball.size)) : 
        ballV[0]*=-1
        setVector((ball.pos[1]+(ball.size[1]/2))-(player1.pos[1]+(player1.size[1]/2)))
    elif(player2.getColBlock(ball.pos, ball.size)) : 
        ballV[0]*=-1
        setVector((ball.pos[1]+(ball.size[1]/2))-(player2.pos[1]+(player2.size[1]/2)))
        resArrays()
    
    if(player1.pos[1]<0):player1.pos[1]=0
    elif(player1.pos[1]+player1.size[1]>400):player1.pos[1]=300
    if(player2.pos[1]<0):player2.pos[1]=0
    elif(player2.pos[1]+player2.size[1]>400):player2.pos[1]=300
    
    
        #Win
    if(ball.pos[0]+ball.size[0]>400):
        ball.pos=[200,200]
        score[0]+=1
        ballV=[-1.0,0.0]
        print("P1:",score[0]," P2:",score[1])
        #Train Model
            #wu
        inputsT=np.vstack(inputs)
        outputsT=np.vstack(outputs)
        print("I:",inputsT)
        print("O:",outputsT)
        model.fit(inputsT,outputsT,callbacks=[cp_callback])
        #Reset Inputs and Outputs
        resArrays()
        player1.pos=[0,150]
        player2.pos=[390,150]
    elif(ball.pos[0]<0):
        ball.pos=[200,200]
        score[1]+=1
        ballV=[1.0,0.0]
        print("P1:",score[0]," P2:",score[1])
        #Reset Inputs and Outputs
        resArrays()
        player1.pos=[0,150]
        player2.pos=[390,150]
    

    #Gae System
    #print(ballV[1])


    #Drawning
    screen.fill(c_off)
    player1.print()
    player2.print()
    ball.print()
    pygame.display.flip()
    clock.tick(60)