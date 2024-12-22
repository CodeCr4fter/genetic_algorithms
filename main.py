import random
import string
import sys
import time
from collections import deque

import numpy as np
import pygame

pygame.init()

WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Genetic Algorithm")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BOX_COLOR = (50, 150, 250)

TEXT_COLOR = (255, 255, 255)
BELT_COLOR = (200, 200, 200)

font = pygame.font.SysFont('Consolas', 38)
POPULATION_SIZE = 1000
MUTATION_RATE = 0.003

def generate_population(charset, size, length):
    return np.random.choice(charset, size=(size, length))



def calc_probabilities(population, target):
    fitnesses = np.sum(population == target, axis=1)
    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:
        return None
    probabilities = fitnesses / total_fitness
    return probabilities


def crossover(parents):
    num_parents, word_length = parents.shape[0], parents.shape[2]
    crossover_points = np.random.randint(1, word_length, size=(num_parents, 1))
    mask = np.arange(word_length) < crossover_points
    parent1, parent2 = parents[:, 0], parents[:, 1]
    children1 = np.where(mask, parent1, parent2)
    children2 = np.where(mask, parent2, parent1)
    return np.vstack((children1, children2))


def mutate(population, mutation_rate, charset):
    mutate_mask = np.random.uniform(size=population.shape) < mutation_rate
    population[mutate_mask] = np.random.choice(charset, size=np.sum(mutate_mask))
    return population
def define_charset(word):
    charset = []
    if any(char.isdigit() for char in word):
        charset += string.digits
    if any(char == char.lower() for char in word):
        charset += string.ascii_lowercase
    if any(char.isspace() for char in word):
        charset += " "
    if any(char == char.upper() for char in word):
        charset += string.ascii_uppercase
    return charset
def genetic_algorithm(target_word):
    word_length = len(target_word)
    charset = define_charset(target_word)
    word_in_np = np.array(list(target_word))

    charset = np.array(charset)
    population = generate_population(charset, POPULATION_SIZE, word_length)
    best_fitness = 0
    while best_fitness < word_length:
        probabilities = calc_probabilities(population, word_in_np)
        parents = population[np.random.choice(population.shape[0], size=(POPULATION_SIZE//2, 2), p=probabilities)]
        new_population = crossover(parents)
        mutated_population = mutate(new_population, MUTATION_RATE, charset)

        population = mutated_population

        best_individual = np.argmax(np.sum(population == word_in_np))
        best_fitness = np.sum(population[best_individual] == word_in_np)
        yield ''.join(population[best_individual]), best_fitness



if __name__ == "__main__":
    input_active = True
    input_text = ""
    best_word = ""
    best_fitness = 0.0

    belt = deque()
    belt_size = 20
    running = True
    ga = None
    generations = 0
    word_length = 0
    while running:
        found = False
        cur_word = ""
        cur_fitness = 0
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if input_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        input_active = False
                        ga = genetic_algorithm(input_text)
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode


        if input_active:
            pygame.draw.rect(screen, BOX_COLOR, (0, HEIGHT // 2 - 50, WIDTH, 100))
            text_surface = font.render(input_text, True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text_surface, text_rect)
        else:
            word_length = len(input_text)
            if ga is not None:
                try:
                    cur_word, cur_fitness = next(ga)
                    generations += 1
                    belt.append(cur_word)
                    if cur_fitness > best_fitness:
                        best_word = cur_word
                        best_fitness = cur_fitness
                    if len(belt) > belt_size:
                        belt.popleft()
                except StopIteration:
                    BOX_COLOR = GREEN
                    found = True
                    running = False

            belt_row_height = HEIGHT // belt_size
            for i, word in enumerate(belt):
                row_rect = pygame.Rect(0, i * belt_row_height, WIDTH, belt_row_height)
                word_surface = font.render(word, True, WHITE)
                word_rect = word_surface.get_rect(center=row_rect.center)
                screen.blit(word_surface, word_rect)
            pygame.draw.rect(screen, BOX_COLOR, (0, HEIGHT // 2 - 50, WIDTH, 100))
            word_surface = font.render(f"Best: {best_word}", True, TEXT_COLOR)
            word_rect = word_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))
            screen.blit(word_surface, word_rect)

            fitness_surface = font.render(f"Generations: {generations}, letters left: {word_length - best_fitness}", True, TEXT_COLOR)
            fitness_rect = fitness_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30))
            screen.blit(fitness_surface, fitness_rect)
        pygame.display.flip()
    if found:
        time.sleep(3)
    pygame.quit()
    sys.exit()
