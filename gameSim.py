import pygame
from pygame.locals import *
import sys
import random #for randomization of the pillars 
import neuralNetwork
import numpy as np 

#class for pillars/incoming structures that we divided into two-> top half and bottom half with a gap between them 
class pillar:
    def __init__(self, x1, x2, y1, y2, width, h1, h2):
        #the width of both the segments is the same 
        self.passed=False #to keep track if the pillar has been passed, since a pillar can only be passed once 
        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2
        self.width=width
        self.h1=h1
        self.h2=h2

    def draw_pillar(self, surface):
        #x,y,width,height
        #drawing top half
        bg_surface=surface 
        pygame.draw.rect(bg_surface, (34, 139, 34), (self.x1, self.y1, self.width, self.h1), width=0, border_radius=0) 
        #drawing bottom half 
        pygame.draw.rect(bg_surface, (34, 139, 34), (self.x2, self.y2, self.width, self.h2), width=0, border_radius=0) 

class bird:
    def __init__(self, x, y, radius, velocity):
        self.x=x
        self.y=y
        self.radius=radius
        self.velocity=velocity #initial velocity 
    
    def drawBird(self, surface):
        pygame.draw.circle(surface, (255, 255, 143) ,(self.x, self.y), self.radius, width=0)

#Helper functions 

def get_state(bird_obj, pillar_obj, gap_centre_y):
    state=np.array([
    [bird_obj.y / 900],
    [bird_obj.velocity / 10],
    [(pillar_obj.x1 - bird_obj.x) / 900],
    [gap_centre_y / 900]
    ])
    return state

def flap(bird_obj):
    #print("Flapping bird..")
    bird_obj.velocity=-3
    pass

def passedGap(bird_obj, pillar):
    if (not pillar.passed and bird_obj.x > pillar.x1 + pillar.width):
        pillar.passed = True
        return True

    return False
    pass

def collided(bird_obj, pillar):
    #respective fitness values/penalties might change 
    screen_height=900
    #collided with top (of window)
    if bird_obj.y - bird_obj.radius <= 0:
        #print("Collided with top of the window!")
        return -10
    #collided with bottom (of window)
    if bird_obj.y + bird_obj.radius >= screen_height:
        #print("Collided with bottom of the window! ")
        return -10
    #collided with top part of pillar (strctures)
    #collided with bottom part of pillar (structures)
    horizontal_overlap = (
        bird_obj.x + bird_obj.radius >= pillar.x1 and
        bird_obj.x - bird_obj.radius <= pillar.x1 + pillar.width
    )
    if horizontal_overlap:
        if bird_obj.y - bird_obj.radius <= pillar.h1:
            #print("Collided with top half!")
            return -10
        if bird_obj.y + bird_obj.radius >= pillar.y2:
            #print("Collided with bottom half!")
            return -10

    return 0


#we also need to put checks for Game Over (Penalty case) and Passing through a gap within a structure (reward)
#for now we will keep the reward 50 and game over -10 
#run the game
def run_game(model):

    #initializing pygame 
    pygame.init()
    start_time=pygame.time.get_ticks() #for stopwatch 
    #screen size
    screen_width=900
    screen_height=900
    screen=pygame.display.set_mode((screen_width, screen_height))

    #window title 
    pygame.display.set_caption("Flappy Bird Neural Network Simulator")
    clock=pygame.time.Clock()
    font=pygame.font.SysFont(None, 40)
    #reference point and background
    bg_surface=pygame.Surface((screen_width, screen_height))
    bg_surface.fill((135, 206, 235))  #blue background
    bg_width = bg_surface.get_width()

    #scrolling variables (to move the screen right to left)
    scroll=0
    scroll_speed=3 #this value willl gradually increaase as time goes on in game 

    #thresholds
    distance_threshold=400
    gap_threshold=70

    #lists needed 
    pillars=[] #we will keep the pillars on screen in this list 

    #pillar data
    gap_size=200
    pillar_width=80

    #bird data
    velocity=0
    gravity=0.05
    bird_radius=20
    #starting positions of the bird 
    bird_x=200
    bird_y=screen_height//2

    #creating bird object 
    bird_obj=bird(bird_x, bird_y, bird_radius, velocity)

    #fitness value 
    fitness=0

    #game loop 
    running=True
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    flap(bird_obj)
        
        
        #randomized new pillar 
        gap_y=random.randint(150, screen_height - 150 - gap_size)
        top_height=gap_y
        bottom_y=gap_y+gap_size
        bottom_height=screen_height-bottom_y

        new_pillar=pillar(
            x1=screen_width,
            x2=screen_width,
            y1=0,
            y2=bottom_y,
            width=pillar_width,
            h1=top_height,
            h2=bottom_height
        )
        gameOver=False 
        reward=1
        #check pillars
        if len(pillars)==0:
            pillars.append(new_pillar)
            #print("Appending a new pillar!")
        else:
            #then we check if the latest pillar is at a certain distance from the right side of the screen 
            last_pillar=pillars[-1]
            if screen_width-last_pillar.x1>=distance_threshold:
                #print("Appending a pillar!")
                pillars.append(new_pillar)

        #scroll logic 
        scroll-=scroll_speed
        if abs(scroll)>bg_width:
            scroll=0

        #moving background tiles 
        screen.blit(bg_surface, (scroll, 0))
        screen.blit(bg_surface, (scroll + bg_width, 0))  

        #both pillars and the bird move relative to the background
        #the only difference is that bird has gravity applied to it 

        #game objects will be rendered here 
        #yellow circle for the bird
        bird_obj.drawBird(screen)

        #green rectangles for the structures with 1 gap in them, gap must always be bigger than the diameter of the bird
        for p in pillars:
            p.draw_pillar(screen)

        #for each pillar there will be teo rectangles one at the top of the frame and one at the bottom of the frame 
        #multiple pillars can exist at a time on a frame granted they have enough threshold distance between them 
        
        #game loop 
        #the structures move from right to left 
        #the gap within a structure is randomized
        #the distance between two structures at a time must be equal or greeater than a threshold 
        #the width of the structures is fixed 

        #neural network integration 
        next_pillar=pillars[0]
        gap_centre_y=next_pillar.h1+gap_size/2
        state=get_state(
            bird_obj,
            next_pillar,
            gap_centre_y
        )
        action=model.action(state)
        #print(action)
        if action==1:
            #print("Model decided to flap.")
            flap(bird_obj)
        else:
            #print("Model decided not to flap.")
            pass

        #physics update
        bird_obj.velocity+=gravity
        bird_obj.y+=bird_obj.velocity

        #calculate reward
        for p in pillars:
            if passedGap(bird_obj, p):
                fitness += 50 
                reward+=5
            #collided funtion returns 0 if no collision has occured
            if (fitness_value := collided(bird_obj, p)):
                fitness+=fitness_value
                reward-=10
                gameOver=True
                break
        

        #get next state 
        next_state=get_state( bird_obj,
            next_pillar,
            gap_centre_y)

        #train the network 
        model.backpropogation(
            state,
            action,
            reward,
            next_state,
            gameOver
        )
        if gameOver:
            return fitness
        #check if the pillar is off screen then we remove it from our list 
        for p in pillars:
            p.x1-=scroll_speed
            p.x2-=scroll_speed
            if p.x1+p.width<0:
                #print("Removing a pillar!")
                pillars.remove(p)
        elapsed_time=(pygame.time.get_ticks() - start_time) // 1000
        score_text=font.render(
                f"Score: {fitness}",
                True,
                (0, 0, 0)
        )
        time_text=font.render(
            f"Time: {elapsed_time}",
            True,
            (0, 0, 0)
        )
        screen.blit(score_text, (700, 20))
        screen.blit(time_text, (700, 60))

        pygame.display.flip()
        clock.tick(0) #FPS
    pygame.quit()
    return None

#main code
#here we make our model and finally start experiencing joy
#let our current state be defined as birds y position, next pillars x and y position and birds veloctiy 
model=neuralNetwork.NeuralNetwork(
    numOfInputs=4,
    numOfOutputs=2,
    hiddenLayers=2,
    numOfNeuronsPerHiddenLayer=[3,4]
)
#initializing weights
model.initialize_weights()
for episode in range(1000):
    fitness=run_game(model)
    print(
        f"Episode {episode} with score {fitness}"
    )