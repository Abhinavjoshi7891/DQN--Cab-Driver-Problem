# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

# {'A':1, 'B':2, 'C':3, 'D':4, 'E':5}
# {'MON':0, 'TUE':1, 'WED':2, 'THU':3, 'FRI':4, 'SAT':5, 'SUN':6}


class CabDriver():

    def __init__(self):
        """
        initialise your state and define your action space and state space
        """
        self.accum_travel_hours = 0
          
        # Locations:       A, B, C, D, E          
        #                  represented by integers 1, 2, 3, 4, 5 (start index 1)
        # Time of the day: 24 hours clock 00:00, 01:00, ..., 22:00, 23:00
        #                  represented by integers 0, 1, 2, 3, 4, ..., 22, 23
        # Day of the week: MON, TUE, WED, THU, FRI, SAT, SUN
        #                  represented by integers 0, 1, 2, 3, 4, 5, 6

          
        # Possible action space = (m-1)*m+1 = 21
        self.action_space = [(1,2), (2,1),
                            (1,3), (3,1),
                            (1,4), (4,1),
                            (1,5), (5,1),
                            (2,3), (3,2),
                            (2,4), (4,2),
                            (2,5), (5,2),
                            (3,4), (4,3),
                            (3,5), (5,3),
                            (4,5), (5,4),
                            (0,0)]
        # Total states (Xi Tj Dk) = 1..m, 1...t, 1...d
        self.state_space = [(x, y, z) for x in range(1, m+1) 
                            for y in range(t) 
                            for z in range(d)]

        # Initialize state to random-state (location, hours, day)
        self.state_init = random.choice(self.state_space)
        
        # Start the first round
        #self.test_run()
    
        self.reset()

    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """
        convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d.
        """
        
        if not state:
            return
        
       
        state_encod = [0 for _ in range(m+t+d)]
        # encode location
        state_encod[state[0] - 1] = 1
        
        # encode hour of the day
        state_encod[m + state[1]] = 1
        
        # encode day of the week
        state_encod[m + t + state[2]] = 1
        
        return state_encod
    


    ## Getting number of requests
    def requests(self, state):
        """
        Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations
        """
        
        # use poisson distribution for generating random # of requests based on average
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)
        
        # limit number of requests to 15
        if requests >15:
            requests = 15

        # (0,0) is not considered as customer request
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]

        if (0, 0) not in actions:
            actions.append((0,0))
            possible_actions_index.append(20)

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """
        Takes in state, action and Time-matrix and returns the reward
        
        reward = (revenue earned from pickup point 𝑝 to drop point 𝑞) - 
                 (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞) - 
                 (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝)        
        """
        cur_loc = state[0]
        st_loc = action[0]
        end_loc = action[1]
        tod = state[1]
        dow = state[2]

        #-----------------
        def get_new_time_day(tod, dow, total_time):
            """
            calculates new time and day
            """
            tod = tod + total_time % (t - 1)
            dow = dow + (total_time // (t - 1))
            
            if tod > (t-1):
                dow = dow + (tod // (t - 1))
                tod = tod % (t - 1)
                if dow > (d - 1):
                    dow = dow % (d - 1)        
            
            return tod, dow
                
        #-----------------
        def get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow):
            """
            calculates the total time of trave based on 
            """
            if not st_loc and not end_loc:
                return 0, 1
            
            t1 = 0
            if st_loc and cur_loc != st_loc:
                t1 = int(Time_matrix[cur_loc-1][st_loc-1][tod][dow])

                # compute new tod and dow after travel t1
                tod, dow = get_new_time_day(tod, dow, t1)
            
            t2 = int(Time_matrix[st_loc-1][end_loc-1][tod][dow])
            #print("t1={} t2={}".format(t1, t2))

            return t1, t2
        #-----------------   
        
        t1, t2 = get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow)
        #print("t1={} t2={}".format(t1, t2))
        
        if not st_loc and not end_loc:
            reward = -C
        else:
            reward = R * t2 - C * (t1 + t2)        
        
        return reward

    

    def next_state_func(self, state, action, Time_matrix):
        """
        Takes state and action as input and returns next state
        """
        cur_loc = state[0]
        st_loc = action[0]
        end_loc = action[1]
        tod = state[1]
        dow = state[2]
        
        #-----------------
        def get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow):
            """
            calculates the total time of trave based on 
            """
            if not st_loc and not end_loc:
                return 1
            
            t1 = 0
            if st_loc and cur_loc != st_loc:
                t1 = int(Time_matrix[cur_loc-1][st_loc-1][tod][dow])
                
                # compute new tod and dow after travel t1
                tod, dow = get_new_time_day(tod, dow, t1)

            t2 = int(Time_matrix[st_loc-1][end_loc-1][tod][dow])
            return t1 + t2

        #-----------------
        def get_new_time_day(tod, dow, total_time):
            """
            calculates new time and day
            """
            tod = tod + total_time % (t - 1)
            dow = dow + (total_time // (t - 1))
            
            if tod > (t-1):
                dow = dow + (tod // (t - 1))
                tod = tod % (t - 1)
                if dow > (d - 1):
                    dow = dow % (d - 1)        
            
            return tod, dow
        #-----------------
        
        total_trv_time = get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow)
        self.accum_travel_hours += total_trv_time
        new_tod, new_dow = get_new_time_day(tod, dow, total_trv_time)
        
        if not st_loc and not end_loc:
            new_loc = state[0]
        else:
            new_loc = action[1]

        return (new_loc, new_tod, new_dow)



    def reset(self):
        self.accum_travel_hours = 0
        self.state_init = random.choice(self.state_space)
        
        return self.action_space, self.state_space, self.state_init

   
        
    
