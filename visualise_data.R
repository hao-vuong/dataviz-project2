library(torch)
library(magick)
library(ggplot2)
library(patchwork)
library(R.utils)

CNN <- nn_module(
  "CNN",
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 32, kernel_size = 3, stride = 1)
    self$conv2 <- nn_conv2d(32, 64, kernel_size = 3, stride = 1)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  forward = function(x) {
    x <- nnf_relu(self$conv1(x))
    x <- nnf_relu(self$conv2(x))
    x <- nnf_max_pool2d(x, kernel_size = 2)
    x <- self$dropout1(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- nnf_relu(self$fc1(x))
    x <- self$dropout2(x)
    x <- self$fc2(x)
    return(x)
  }
)

model <- CNN()
model_path <- "mnist_cnn_model.pth"

if (file.exists(model_path)) {
  model$load_state_dict(torch_load(model_path))
  model$eval()
  cat("Model loaded successfully!\n")
} else {
  cat("Error: mnist_cnn_model.pth not found. Please ensure the model file is in the correct directory.\n")
}

image_to_tensor_transform <- function(image_path_or_magick_image) {
  img <- NULL
  if (is.character(image_path_or_magick_image)) {
    img <- image_read(image_path_or_magick_image)
  } else if ("magick-image" %in% class(image_path_or_magick_image)) {
    img <- image_path_or_magick_image
  } else {
    stop("Input must be a file path or a magick image object.")
  }

  img_gray <- image_convert(img, type = "Grayscale")
  img_resized <- image_resize(img_gray, "28x28!")
  img_array <- as.integer(img_resized[[1]])
  tensor <- torch_tensor(img_array / 255, dtype = torch_float())
  tensor <- tensor$unsqueeze(1)
  tensor <- tensor$permute(c(1, 2, 3))
  return(tensor)
}

get_feature_map_image_r <- function(model, image_tensor, selected_maps = 6) {
  activations <- list()

  hook_fn_conv1 <- function(module, input, output) {
    activations[["conv1"]] <<- output$detach()
  }
  hook_fn_conv2 <- function(module, input, output) {
    activations[["conv2"]] <<- output$detach()
  }

  hook1 <- model$conv1$register_forward_hook(hook_fn_conv1)
  hook2 <- model$conv2$register_forward_hook(hook_fn_conv2)

  with_no_grad({
    model(image_tensor$unsqueeze(1))
  })

  hook1$remove()
  hook2$remove()

  plot_list <- list()

  for (name in names(activations)) {
    act_tensor <- activations[[name]]
    if (act_tensor$ndim == 3) {
      act_tensor <- act_tensor$unsqueeze(1)
    }

    num_channels <- act_tensor$size(2)
    indices_to_plot <- as.integer(round(seq(1, num_channels, length.out = selected_maps)))

    for (i in seq_along(indices_to_plot)) {
      idx <- indices_to_plot[i]
      feature_map_matrix <- as.array(act_tensor[1, idx, , ])
      df <- expand.grid(y = 1:nrow(feature_map_matrix), x = 1:ncol(feature_map_matrix))
      df$value <- as.vector(feature_map_matrix)

      p <- ggplot(df, aes(x = x, y = y, fill = value)) +
        geom_tile() +
        scale_fill_viridis_c() +
        theme_void() +
        theme(legend.position = "none", plot.title = element_text(size = 8, hjust = 0.5)) +
        ggtitle(paste0(name, " [", idx - 1, "]"))

      plot_list[[length(plot_list) + 1]] <- p
    }
  }

  combined_plot <- wrap_plots(plot_list, nrow = 2, ncol = selected_maps)
  return(combined_plot)
}

dummy_image_path <- "test_digit.png"

if (!file.exists(dummy_image_path)) {
  img_dummy <- image_blank(28, 28, color = "black")
  img_dummy <- image_draw(img_dummy)
  image_write(img_dummy, path = dummy_image_path, format = "png")
  cat("Created a dummy image for demonstration:", dummy_image_path, "\n")
}

if (file.exists(dummy_image_path)) {
  input_tensor <- image_to_tensor_transform(dummy_image_path)
  cat("Input tensor shape:", paste(input_tensor$shape, collapse = "x"), "\n")

  with_no_grad({
    output <- model(input_tensor$unsqueeze(1))
  })
  prediction <- output$argmax(dim = 2)$item()
  cat("Predicted digit:", prediction, "\n")

  feature_map_plot <- get_feature_map_image_r(model, input_tensor, selected_maps = 6)
  print(feature_map_plot)
} else {
  cat("Cannot run example: Dummy image file not found. Please create one or provide a valid image path.\n")
}
