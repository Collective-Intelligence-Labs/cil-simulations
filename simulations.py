from mesa import Agent, Model, datacollection
from mesa.time import RandomActivation
import random
import uuid
import math
import matplotlib.pyplot as plt
from mesa.space import ContinuousSpace
import matplotlib.patches as patches
from scipy import stats
import numpy as np

class Review:
    def __init__(self, creator_id , project_id, score):
        self.id = uuid.uuid4()
        self.creator_id = creator_id
        self.project_id = project_id
        self.score = score

    @staticmethod
    def validate_score(score):
        return score >= 1 and score <= 5

class Project:
    def __init__(self, creator_id, size, pos, model):
        self.id = uuid.uuid4()
        self.creator_id = creator_id
        self.size = size
        self.radius = size
        self.pos = pos
        self.model = model
        self.balance = 0
        self.review_step = 0
        self.status = 'active'  # New attribute to store the project status
        self.contribution = 0   # New attribute to store the total contribution
        self.reviewed_contribution = 0 

    # New method to handle contributions
    def add_contribution(self, contribution):
        self.contribution += contribution
        if self.contribution >= self.size:
            self.finilize()
    
    def finilize(self):
        self.contribution = self.size
        self.impact = self.model.calculate_project_impact_area(self)
        self.status = 'finalized'

    def start_review(self):
        #Enable review period and freeze the public/available state of the project for review
        self.model.projects_in_review.append(self)
        self.reviewed_contribution = self.contribution
        self.status = 'review'
    
    def end_review(self):
        self.status = 'active'
    
    def calculate_area(self):
        return math.pi * self.size ** 2

    def calculate_intersection_area(self, other):
        d = np.linalg.norm(np.array(self.pos) - np.array(other.pos))

        # If the projects do not intersect, return 0
        if d >= self.radius + other.radius:
            return 0

        # Calculate the area of intersection
        intersection_area = (
            self.radius ** 2 * math.acos((d ** 2 + self.radius ** 2 - other.radius ** 2) / (2 * d * self.radius)) +
            other.radius ** 2 * math.acos((d ** 2 + other.radius ** 2 - self.radius ** 2) / (2 * d * other.radius)) -
            0.5 * math.sqrt((-d + self.radius + other.radius) * (d + self.radius - other.radius) * (d - self.radius + other.radius) * (d + self.radius + other.radius))
        )

        return intersection_area

MAX_ENERGY_REFILL = 0.2
MODEL_SIZE = 50
MIN_AWARENESS = 1
MAX_AWARENESS = math.sqrt(MODEL_SIZE)
NUM_AGENTS = 100
STEPS = 500
INITIAL_PROJECTS = 2
ERROR_DELTA_MAX = 0.1

class SingularityAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.energy = 1
        self.expertise = self.initiate_with_gauss(1.0, 0.3, 0.6, 2.0)
        self.talent = self.initiate_with_gauss(1.0, 0.3, 0.6, 2.0)
        self.projects = []
        self.balance = self.initiate_with_gauss(100.0, 30, 0)
        self.rate = self.initiate_with_gauss(10.0, 3, 1)
        self.awareness = max(MIN_AWARENESS, math.sqrt(self.expertise))
        self.explored_projects = []
        self.exploation_status = {}

    def set_expertise(self, expertise):
        self.model.update_agent_expertise(self.expertise, expertise)
        self.expertise = expertise
        self.awareness = min(MAX_AWARENESS,max(MIN_AWARENESS, math.sqrt(self.expertise)))
    
    def select_action(self):
        action_type = random.choice(["rest", "work"])

        if action_type == "rest":
            self.rest()
        elif action_type == "work":
            work_type = random.choice(["create", "learn"])

            if work_type == "create":
                create_type = random.choice(["new knowledge", "review"])
                
                if create_type == "new knowledge":
                    self.create_new_knowledge()
                else:
                    self.review_creation()

            elif work_type == "learn":
                self.learn()

    @staticmethod
    def initiate_with_gauss(mean, std_dev, min_val=None, max_val=None):
        result = random.gauss(mean, std_dev)
        if min_val is not None:
            result = max(result, min_val)
        if max_val is not None:
            result = min(result, max_val)
        return result

    def rest(self):
        self.refill_energy()

    def refill_energy(self):
        energy_refill = self.random.uniform(0, MAX_ENERGY_REFILL)
        self.energy = min(1, self.energy + energy_refill)

    def propose_project_pos(self):
        # Determine the project coordinates within the sphere of the agent's knowledge
        x_coord = self.pos[0] + random.uniform(-self.expertise, self.expertise)
        y_coord = self.pos[1] + random.uniform(-self.expertise, self.expertise)
        
        coordinates = (x_coord, y_coord)
        return coordinates

    def create_new_project(self):
        targeted_size = math.sqrt(self.expertise) * self.talent * random.uniform(0.5, 1.5)  # This will give a size relative to the agent's expertise
     
        coordinates = self.propose_project_pos()
        while not self.is_inside_circle(coordinates):
            coordinates = self.propose_project_pos()

        # Create a new project with the determined attributes
        project = Project(self.unique_id, targeted_size, coordinates, self.model)
        self.projects.append(project)
        self.model.projects.append(project)
        return project  # Return the created project

    def create_new_knowledge(self):
        # Select one of the active projects to contribute to
        active_projects = [p for p in self.projects if p.status == 'active']

        action = random.choices(
            population=["create_new_project", "contribute_to_existing"],
            weights=[0.1, 0.9],  # 20/80 chance for each action
            k=1
        )[0]
        project = None
        if action == "contribute_to_existing" and active_projects:
            project = random.choice(active_projects)
        elif action == "create_new_project":
            project = self.create_new_project()

        energy_consumed = random.uniform(0, self.energy)  # Random energy consumption between 0 and current energy level
        self.energy -= energy_consumed  # Deduct energy consumed from agent's current energy

        # Determine the size of the contribution 
        contribution = math.sqrt(math.sqrt(self.expertise)) * self.talent * energy_consumed
        # Update the project with the contribution
        
        if project:   
            project.add_contribution(contribution)
            self.set_expertise(self.expertise + contribution)
            if random.random() < 0.1:
                project.start_review()
        
#TODO:
#1. knowledge creation impact should be calcualte based on the filled area taking into account that there might be intercentions with other projects and this volume should be reduced than        
#2. there should be also taking into account that the close to the center to higher density (complexity) of knowldge creation therefore it would affect the size and the amount of contribution
#3. sometime to more deeper you need more kmowdlge from other side and it also should depend somehow in the agent awareness
#4. maybe we should also emulate focus - when the focus of attention stays long enough on a specific segment the quality of contibution growth or the enerhy spent declines (bssically becoming profficient) with some max benefit from focus (1 for example)


    def review_creation(self):
        # Implement review creation logic
        if (len(self.explored_projects) == 0):
            return
        
        review_projects = [p for p in self.explored_projects if p.status == 'review']
        if (len(review_projects) == 0):
            return
        project = random.choice(review_projects)
        expertise_percentile = self.model.calculate_expertise_percentile(self.expertise)
        delta = ERROR_DELTA_MAX / expertise_percentile
        err = random.uniform(-delta/2, delta/2)
        project_percentile = self.model.calculate_impact_percentile(project.reviewed_contribution)
        guessed = min(0, max(project_percentile + err, 1))
        score = self.get_score_from_percent(guessed)
        print(expertise_percentile)

    def get_score_from_percent(self, percent):
        if percent <= 0.2:
            return 1
        elif percent <= 0.4:
            return 2
        elif percent <= 0.6:
            return 3
        elif percent <= 0.8:
            return 4
        else:
            return 5

    def is_inside_circle(self, position):
        # assuming the center of the circle is (0, 0)
        circle_radius = 50  # adjust the radius according to your needs
        x, y = position
        return x**2 + y**2 <= circle_radius**2
    
    def learn(self):
        energy_consumed = random.uniform(0, self.energy)
        projects = self.search_projects_in_sphere()
        if (len(projects) == 0):
            return
        project = random.choice(projects)
        if (project.id not in self.exploation_status):
            self.exploation_status[project.id] = 0
        if (self.exploation_status[project.id] >= project.contribution):
            return
        gained_expertise = self.talent * energy_consumed * math.sqrt(math.sqrt(project.contribution))
        self.set_expertise(self.expertise + gained_expertise)
        self.exploation_status[project.id] += gained_expertise
        self.explored_projects.append(project)

    def move(self):
        max_distance = math.sqrt(self.awareness) * self.energy
        
        # Choose a random direction
        direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        
        # Choose a random step size up to the maximum
        step_size = random.uniform(0, max_distance)
        
        # Calculate the new position
        new_position = (self.pos[0] + direction[0] * step_size, self.pos[1] + direction[1] * step_size)
        
        if self.is_inside_circle(new_position):
            self.pos = new_position
            self.energy -= step_size / max_distance
            self.energy = max(0, self.energy)

    def consume_energy(self):
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def step(self):
        if random.random() < 0.5:  # 50% chance to move
            self.move()
        self.refill_energy()
        self.consume_energy()
        self.select_action()

    def search_projects_in_sphere(self):
        projects_in_sphere = []
        for project in self.model.projects:
            if self.distance_to(project) <= self.awareness:
                projects_in_sphere.append(project)
        return projects_in_sphere

    def distance_to(self, project):
        return ((self.pos[0] - project.pos[0]) ** 2 + (self.pos[1] - project.pos[1]) ** 2) ** 0.5    


class SingularityModel(Model):
    def __init__(self, num_agents, R, number_of_projects):
        self.schedule = RandomActivation(self)
        self.projects = []
        self.projects_in_review = []
        self.radius = R
        self.expertise_mean = 0
        self.expertise_variance = 0
        self.impact_mean = 0
        self.impact_variance = 0
        self.num_agents = 0
        self.space = ContinuousSpace(self.radius, self.radius, True, -self.radius, -self.radius )
        self.datacollector = datacollection.DataCollector(
            model_reporters={"AgentCount": lambda m: m.schedule.get_agent_count(),"ProjectsCount": lambda m: len(m.projects), "Expertise Mean": lambda m: m.expertise_mean},
            agent_reporters={"Energy": lambda a: a.energy, "Projects": lambda a: len(a.projects), "Expertise": lambda a: a.expertise, "x": lambda a: a.pos[0], "y": lambda a: a.pos[1], "awareness": lambda a: a.awareness}
        )

        for i in range(num_agents):
            a = SingularityAgent(i, self)
            self.schedule.add(a)
            self.space.place_agent(a, self.random_point_in_space())
            self.add_agent_expertise(a.expertise)
        
        for i in range(number_of_projects): # define number_of_projects based on your need
        
            size = self.random.uniform(1, math.sqrt(R))  # define min and max size
            project = Project(None, size, self.random_point_in_space(), self)
            project.finilize()
            self.projects.append(project)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.finilize_reviews()

    def finilize_reviews(self):
        print(len(self.projects_in_review))
        finilized = []
        for project in self.projects_in_review:
            project.review_step += 1
            if (project.review_step == 5):
                finilized.append(project)
                project.end_review()
        for remove in finilized:
            self.projects_in_review.remove(remove)

    def random_point_in_space(self):
         # Generate random angle and radius
            angle = 2 * math.pi * self.random.random()  # Random angle
            rad = self.radius * math.sqrt(self.random.random())  # Random radius

            # Convert polar coordinates to cartesian
            x = rad * math.cos(angle)
            y = rad * math.sin(angle)
            return (x,y)
    
    def add_agent_expertise(self, expertise):
        self.num_agents += 1
        delta = expertise - self.expertise_mean
        self.expertise_variance = (self.expertise_variance * (self.num_agents - 1) + delta * delta) / self.num_agents
        self.expertise_mean += delta / self.num_agents

    def remove_agent_expertise(self, expertise):
        if self.num_agents > 1:
            self.num_agents -= 1
            delta = expertise - self.expertise_mean
            self.expertise_variance = (self.expertise_variance * (self.num_agents + 1) - delta * delta) / self.num_agents
            self.expertise_mean -= delta / self.num_agents
        else:
            self.expertise_mean = 0
            self.expertise_variance = 0

    def update_agent_expertise(self, old_expertise, new_expertise):
        # Temporarily remove the old expertise
        self.remove_agent_expertise(old_expertise)

        # Then add the new expertise
        self.add_agent_expertise(new_expertise)
    
    def calculate_expertise_percentile(self, expertise):
        return self.calculate_percentile(expertise, self.expertise_mean, self.expertise_variance)
    
    def calculate_impact_percentile(self, impact):
        return self.calculate_percentile(impact, self.expertise_mean, self.expertise_variance)

    @staticmethod
    def calculate_percentile(value, mean, variance):
        # Calculate the percentile of the given expertise value based on the current distribution
        if variance < 0:
            raise ValueError("Variance must be non-negative")
        std_dev = math.sqrt(variance)
        if std_dev > 0:
            z_score = (value - mean) / std_dev
            return stats.norm.cdf(z_score)
        else:
            return 0.5 if value == mean else (1 if value > mean else 0)

    def calculate_project_impact_area(self, project):
        # Start with the total area of the project
        area = project.calculate_area()

        # Subtract the areas of intersection with other projects
        for other_project in self.projects:
            if other_project is not project and other_project.status == "finilized":  # Don't calculate intersection with itself
                intersection_area = project.calculate_intersection_area(other_project)
                area -= intersection_area

        return area

R = MODEL_SIZE
# Create model object and run
model = SingularityModel(NUM_AGENTS, R, INITIAL_PROJECTS)

# Get the agent data.
#agent_data = model.datacollector.get_agent_vars_dataframe()

fig, ax = plt.subplots()

# Set equal aspect and the x and y limits
ax.set_aspect('equal', 'box')
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)

# Add a circle of radius R to represent the field
circle = plt.Circle((0, 0), R, edgecolor='black', facecolor='None', linewidth=1)
ax.add_patch(circle)

for i in range(STEPS):
    ax.cla()
    # Set equal aspect and the x and y limits
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

    frame_num = i
    model.step()
    
    circle = plt.Circle((0, 0), R, edgecolor='black', facecolor='white', linewidth=1)
    ax.add_patch(circle)
  
    # Draw all the projects
    for project in model.projects:
        project_circle = patches.Circle(project.pos, project.size, color='red', alpha=min(project.contribution/project.size, 1))
        ax.add_artist(project_circle)

    # Draw all the agents
    for agent in model.schedule.agents:
        agent_circle = patches.Circle(agent.pos, agent.awareness, fill=False)
        ax.add_artist(agent_circle)
    

    # Pause for a bit to create an animation effect
    plt.pause(0.1)



#ani = animation.FuncAnimation(fig, update, frames=range(len(model_data)), repeat=False)

plt.show()


# Get the data collected during the simulation.
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()

print(model_data)
print(agent_data)

print(model_data.describe())
print(agent_data.describe())
