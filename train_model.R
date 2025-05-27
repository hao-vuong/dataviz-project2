# Packages ----------------------------------------------------------------
library(torch)
library(torchvision)
library(magrittr) # For the pipe operator %>%
library(progress) # For the progress bar

# Define the transformation pipeline --------------------------------------
transform <- . %>%
  transform_to_tensor() %>%
  transform_normalize(mean = c(0.1307), std = c(0.3081))

# Datasets and loaders ----------------------------------------------------
train_dataset <- mnist_dataset(
  root = "./data", # Path to download/load data
  train = TRUE,    # Use training set
  transform = transform, # Apply the defined transform
  download = TRUE  # Download if not present
)

test_dataset <- mnist_dataset(
  root = "./data",
  train = FALSE,   # Use test set
  transform = transform,
  download = TRUE
)

# Dataloaders
train_dl <- dataloader(
  train_dataset,
  batch_size = 128,
  shuffle = TRUE,    # Shuffle training data
  drop_last = TRUE   # Drop the last incomplete batch if dataset size is not divisible by batch_size
)

test_dl <- dataloader(
  test_dataset,
  batch_size = 128,
  shuffle = FALSE,   # No need to shuffle test data
  drop_last = TRUE
)

# Building the network ---------------------------------------------------
net <- nn_module(
  "Net",
  initialize = function() {
    # Convolutional layers
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1)
    
    # Dropout layers
    self$dropout1 <- nn_dropout2d(p = 0.25) # For 2D feature maps
    self$dropout2 <- nn_dropout(p = 0.5)    # For 1D features (after flattening)
    
    # Fully connected layers
    # Input features to fc1: 64 channels * 12 height * 12 width = 9216
    # MNIST images are 28x28.
    # After conv1 (kernel 3, stride 1, no padding): 28 - 3 + 1 = 26x26
    # After conv2 (kernel 3, stride 1, no padding): 26 - 3 + 1 = 24x24
    # After max_pool2d (kernel 2): 24 / 2 = 12x12
    # So, flattened size is 64 channels * 12 * 12 = 9216. This is correct.
    self$fc1 <- nn_linear(in_features = 9216, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 10) # 10 classes for MNIST
  },
  
  forward = function(x) {
    # Apply conv1 -> relu
    x <- self$conv1(x)
    x <- nnf_relu(x)
    
    # Apply conv2 -> relu
    x <- self$conv2(x)
    x <- nnf_relu(x)
    
    # Apply max pooling
    x <- nnf_max_pool2d(x, kernel_size = 2)
    
    # Apply first dropout (on 2D feature maps)
    x <- self$dropout1(x)
    
    # Flatten the tensor for the fully connected layers
    # start_dim = 2 means flatten from the first channel dimension onwards (batch_size, features)
    x <- torch_flatten(x, start_dim = 2) 
    
    # Apply fc1 -> relu
    x <- self$fc1(x)
    x <- nnf_relu(x)
    
    # Apply second dropout (on 1D features)
    x <- self$dropout2(x)
    
    # Output layer (logits)
    # nnf_cross_entropy expects raw logits
    output <- self$fc2(x)
    
    return(output)
  }
)

model <- net()

# Move model to selected device (CPU or CUDA if available)
device <- if (cuda_is_available()) "cuda" else "cpu"
model$to(device = device)
cat("Using device:", device, "\n")

# Training loop -----------------------------------------------------------
optimizer <- optim_sgd(model$parameters, lr = 0.01) # Stochastic Gradient Descent optimizer

epochs <- 10 # Number of epochs to train for

for (epoch in 1:epochs) {
  
  # Progress bar for training
  # Corrected format string using paste0
  pb_train <- progress::progress_bar$new(
    total = length(train_dl),
    format = paste0("Epoch ", epoch, "/", epochs, " [:bar] :percent eta: :eta Loss: :loss")
  )
  
  train_losses <- c() # To store loss for each batch in training
  test_losses <- c()  # To store loss for each batch in testing
  
  # --- Training phase ---
  model$train() # Set the model to training mode (enables dropout, etc.)
  
  # Iterate over training dataloader
  coro::loop(for (b in train_dl) {
    optimizer$zero_grad() # Clear gradients from previous step
    
    # Move data to the selected device
    # b[[1]] is the batch of images, b[[2]] is the batch of labels
    input_tensor <- b[[1]]$to(device = device)
    target_tensor <- b[[2]]$to(device = device)
    
    # Forward pass: compute model output
    output <- model(input_tensor)
    
    # Calculate loss
    loss <- nnf_cross_entropy(output, target_tensor)
    
    # Backward pass: compute gradients
    loss$backward()
    
    # Update model parameters
    optimizer$step()
    
    train_losses <- c(train_losses, loss$item()) # Store loss value (as a plain R number)
    pb_train$tick(tokens = list(loss = mean(train_losses, na.rm = TRUE))) # Update progress bar
  })
  
  # --- Evaluation phase ---
  model$eval() # Set the model to evaluation mode (disables dropout, etc.)
  
  # Disable gradient calculations for evaluation (saves memory and computation)
  with_no_grad({ 
    coro::loop(for (b in test_dl) {
      # Move data to the selected device
      input_tensor <- b[[1]]$to(device = device)
      target_tensor <- b[[2]]$to(device = device)
      
      # Forward pass
      output <- model(input_tensor)
      
      # Calculate loss
      loss <- nnf_cross_entropy(output, target_tensor)
      test_losses <- c(test_losses, loss$item())
    })
  })
  
  # Print epoch summary
  cat(sprintf("\nEpoch %d/%d Completed: [Train Loss: %.4f] [Test Loss: %.4f]\n",
              epoch, epochs, mean(train_losses, na.rm = TRUE), mean(test_losses, na.rm = TRUE)))
}

cat("Training finished.\n")

# Example: Save the trained model

# Final evaluation on the test set after all epochs (Loss and Accuracy)
model$eval()
final_test_losses <- c()
final_correct_predictions <- 0
final_total_evaluated <- 0
cat("\nStarting final evaluation on the test set...\n")
pb_final_eval <- progress::progress_bar$new(
  total = length(test_dl),
  format = "Final Evaluation [:bar] :percent eta: :eta Loss: :loss Acc: :acc"
)

with_no_grad({
  coro::loop(for (b in test_dl) {
    input_tensor <- b[[1]]$to(device = device)
    target_tensor <- b[[2]]$to(device = device)
    output <- model(input_tensor)
    loss <- nnf_cross_entropy(output, target_tensor)
    final_test_losses <- c(final_test_losses, loss$item())
    
    predicted_classes <- torch_argmax(output, dim = 2)
    final_correct_predictions <- final_correct_predictions + sum(predicted_classes == target_tensor)$item()
    final_total_evaluated <- final_total_evaluated + target_tensor$size(1)
    
    current_acc <- if (final_total_evaluated > 0) final_correct_predictions / final_total_evaluated else 0
    pb_final_eval$tick(tokens = list(loss = mean(final_test_losses, na.rm = TRUE), acc = sprintf("%.4f", current_acc)))
  })
})

final_overall_accuracy <- if (final_total_evaluated > 0) final_correct_predictions / final_total_evaluated else 0
final_overall_loss <- mean(final_test_losses, na.rm = TRUE)

cat(sprintf("\nFinal Model Performance on Test Set: [Loss: %.4f] [Accuracy: %.4f]\n",
            final_overall_loss, final_overall_accuracy))

torch_save(model, "mnist_r_model.pt")
# To load: model <- torch_load("mnist_r_model.pt")