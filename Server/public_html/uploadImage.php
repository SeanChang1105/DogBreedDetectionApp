<?php
  $image = $_POST["image"];

  if(file_put_contents("/data/lam138/DogImage/dog.jpg", base64_decode($image))) {
    echo "Image Uploaded\n";
  }
  else {
    echo "Image Failed To Upload\n";
  }
?>