using Random

# Set a specific seed (replace 42 with your desired seed)
seed_value = 42
Random.seed!(seed_value)

# Generate a random number
random_number = rand()

println("Random Number: ", random_number)