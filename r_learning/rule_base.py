
import random
import torch

class SnakeRuleBase:
    def __init__(self, boundary):
        self.bx, self.by = boundary

        self.directions = [
            (0, -1), # 0, up
            (0, 1), # 1, down
            (-1, 0), # 2, left
            (1, 0), # 3, right
            (1, -1), # 4, right, up
            (1, 1), # 5, right down
            (-1, -1), # 6, left, up
            (-1, 1) # 7, left, down
        ]


        self.action_map_dir = [
            [0, 2, 3, 4, 6],
            [1, 2, 3, 5, 7],
            [0, 1, 2, 6, 7],
            [0, 1, 3, 4, 5]
        ]

        self.dir_action = [
            0,
            1,
            2,
            3,
            3,
            3,
            2,
            2
        ]

        self.escape_dir = [
            [2, 3],
            [2, 3],
            [0, 1],
            [0, 1],
            [2],
            [0],
            [1],
            [0]
        ]
        self.device = 'cuda'
        self.num_actions = 4
        self.trust_ = 0


    def select_action_by_rule(self, info):

        cmd_map = {
            "UP" : 0,
            "DOWN" : 1,
            "LEFT" : 2,
            "RIGHT" : 3,
            "NONE" : 4
        }
        cur_action = info['action']
        cur_action = cmd_map[cur_action]
        dir_info = []
        hx, hy = info['snake_head']
        fx, fy = info['food']
        vec = (fx - hx), (fy - hy)
        fvec = (fx - hx), (fy - hy)
        for dir in self.directions:
            apple, tail, body, wall = self.lookInDirection(dir, info)
            dir_info.append((apple, tail, body, wall))
        
        candicate = []
        for dir in self.action_map_dir[cur_action]:
            apple, tail, body, wall = dir_info[dir]
            # print(wall)
            # print(apple, tail, body, wall)

            if wall == 1 or body == 1 or tail == 1:
                self.trust_ = 0

            if tail != 1 and body != 1 and apple and wall != 1:
                self.trust_ = 1
                return torch.tensor([[self.dir_action[dir]]], device=self.device,
                    dtype=torch.long)
            elif tail != 1 and body != 1 and not apple:
                if self.dot(vec, self.directions[dir]) > 0 and wall != 1:
                    candicate.append(self.dir_action[dir])
                elif wall == 0:
                    candicate.append(self.dir_action[dir])


        
        if self.trust_ == 1:
            return torch.tensor([[cur_action]], device=self.device,
                    dtype=torch.long)

        if len(candicate) == 0:
            for ed in self.escape_dir[dir]:
                edir = self.directions[ed]
                if self.dot(edir, fvec) > 0:
                    candicate.append(ed)
            # candicate.extend(self.escape_dir[dir])
            # a = torch.tensor([[cur_action]], device=self.device,
            #         dtype=torch.long)
            # print(f"Random: {a}")
            # return a

        # print(candicate)
    
        action = random.sample(candicate, 1)[0]
        a = torch.tensor([[action]], device=self.device,
                    dtype=torch.long)
        # print(f'Candicate : {a}')
        return a
    
    def dot(self, x, y):
        x1, x2 = x
        y1, y2 = y

        return x1 * y1 + x2 * y2

    def collideWall(self, x, y):

        collide = False
        collide = True if x >= 300 or x < 0 else False
        collide = True if y >= 300 or y < 0 else collide

        return collide

    def equal(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2 and y1 == y2:
            return True
        return False

    def lookInDirection(self, dir, info):

        hx, hy = info['snake_head']
        fx, fy = info['food']
        tx, ty = info['snake_tail']
        bodies = info['snake_body']

        dx, dy = dir
        cx, cy = hx, hy
        distance = 0
        apple = False
        apple_dis = -1
        tail = False
        tail_dis = -1
        body_flag = False
        body_dis = -1

        # dis food
        vec = ((fx- hx), (fy-hy))
        apple = bool(self.is_parallel(vec, dir))

        # dis tail

        vec = ((tx-hx), (ty-hy))
        tail = self.is_parallel(vec, dir) * self.vec_len(vec)

        # dis body
        body_flag = 1000
        for bx, by in bodies:
            vec = ((bx - hx), (by - hy))
            tmp_flag = self.is_parallel(vec, dir) * self.vec_len(vec)
            if tmp_flag < body_flag:
                body_flag = tmp_flag


        # dis wall
        cx += dx*10
        cy += dy*10
        if self.collideWall(cx, cy):
            wall_dis = 1
        else:
            wall_dis = 0

        # while not self.collideWall(cx, cy):

        #     cx += dx*10
        #     cy += dy*10
        #     distance += 1

        #     # Check if there is an apple
        #     if not apple:
        #         apple = apple or self.equal((cx, cy), (fx, fy))
        #         apple_dis = distance if apple else apple_dis

        #     # check tail
        #     if not tail:
        #         tail = tail or self.equal((cx, cy), (tx, ty))
        #         tail_dis = distance if tail else tail_dis


        #     # Check body
        #     if not body_flag:
        #         for body in bodies:
        #             bx, by = body
        #             body_flag = body_flag or self.equal((cx, cy), (bx, by))
        #             body_dis = distance if body_flag else body_dis

        
        if body_flag:
            apple = False
            
        # wall_dis = 1/(distance+1e-20)


        return apple, tail, body_flag, wall_dis

    def is_parallel(self, x, y):

        x_len = self.vec_len(x)
        y_len = self.vec_len(y)

        if x_len == 0 or y_len == 0:
            return 1
        
        cos = self.dot(x, y)** 2 / (self.vec_len(x) * self.vec_len(y)) 

        # print(cos)
        return 1 if int(cos) == 1 else 0
    
    def vec_len(self, x):
        x1, x2 = x
        return x1**2 + x2**2
        

        

            


        

            

                

        





        
        