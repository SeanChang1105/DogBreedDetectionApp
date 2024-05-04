package com.example.dogidentification;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.canhub.cropper.CropImageView;

public class MainActivity extends AppCompatActivity {
    public static int if_gallery = 0;

    private ActivityResultLauncher<Intent> cameraLauncher;
    //private ActivityResultLauncher<Intent> cropLauncher;
    private ActivityResultLauncher<Intent> galleryLauncher;
    //String serverUrl = "https://bluepill.ecn.purdue.edu/~lam138/uploadImage.php";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button camera_open_id = findViewById(R.id.camera_button);
        Button galleryButton = findViewById(R.id.gallery_button);

        cameraLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == RESULT_OK) {
                if_gallery = 0;
                //bitmap: a digital image represented as a grid of pixels in a two-dimensional coordinate system, where each pixel contains color information
                Bitmap photo = (Bitmap) result.getData().getExtras().get("data");
                showConfirmationDialog(photo);
            }
        });

        galleryLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                        // Get the URI of the selected file
                        Uri uri = result.getData().getData();
                        try {
                            // Convert Uri to Bitmap
                            Bitmap selectedPhoto = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));;
                            AlertDialog.Builder helloBuilder = new AlertDialog.Builder(this);
                            helloBuilder.setTitle("Hello");
                            helloBuilder.setIcon(new BitmapDrawable(getResources(), selectedPhoto)); // Set bitmap as icon
                            helloBuilder.setMessage("Do you want to use this photo?");
                            helloBuilder.setPositiveButton("Yes", (dialog, which) -> {
                                if_gallery = 1;
                                //showConfirmationDialog_gallery(selectedPhoto);
                                showConfirmationDialog_gallery(uri);
                            });
                            helloBuilder.setNegativeButton("No", (dialog, which) -> {
                                openGallery();
                            });
                            helloBuilder.show();
                            //showConfirmationDialog(selectedPhoto);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });

        camera_open_id.setOnClickListener(v -> openCamera());
        galleryButton.setOnClickListener(v -> openGallery());
    }


    private void openGallery() {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        photoPickerIntent.setAction(Intent.ACTION_GET_CONTENT);
        galleryLauncher.launch(photoPickerIntent);
    }


    private void openCamera() {
        Intent camera_intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraLauncher.launch(camera_intent);
    }

    // for take a photo
    private void showConfirmationDialog(Bitmap photo) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Confirm");
        builder.setMessage("Do you want to use this photo?");
        builder.setPositiveButton("Yes", (dialog, which) -> {
            // Initiate image cropping
            /*CropImageView cropImageView = findViewById(R.id.cropImageView);
            cropImageView.setImageBitmap(photo);
            Bitmap cropped = cropImageView.getCroppedImage();
            sendImgToServer(cropped);*/
            try {
                Intent uploadPage = new Intent(this, Processing.class);
                uploadPage.putExtra("image", photo);
                startActivity(uploadPage);
            } catch (Exception e) {
                AlertDialog.Builder builder2 = new AlertDialog.Builder(this);
                builder2.setTitle("error");
                builder2.setMessage("error");
                builder2.show();
                throw new RuntimeException(e);
            }

        });
        builder.setNegativeButton("No", (dialog, which) -> {
            // Retake photo
            openCamera();
        });
        builder.show();
    }

    // for select photo from gallery
    private void showConfirmationDialog_gallery(Uri uri) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Confirm");
        builder.setMessage("Do you want to use this photo?");
        builder.setPositiveButton("Yes", (dialog, which) -> {
            try {
                Intent uploadPage = new Intent(this, Processing.class);
                uploadPage.putExtra("imageUri", uri.toString()); // Pass the URI as a string
                startActivity(uploadPage);
            } catch (Exception e) {
                AlertDialog.Builder builder2 = new AlertDialog.Builder(this);
                builder2.setTitle("error");
                builder2.setMessage("error");
                builder2.show();
                throw new RuntimeException(e);
            }

        });
        builder.setNegativeButton("No", (dialog, which) -> {
            // Reopen gallery
            openGallery();
        });
        builder.show();
    }


    /*private void sendImgToServer(Bitmap croppedBitmap) {
        StringRequest stringRequest = new StringRequest(Request.Method.POST, serverUrl,
                response -> {
                    Toast.makeText(getApplicationContext(), response, Toast.LENGTH_LONG).show();
                    AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                    builder.setTitle("Sent Successfully");
                    builder.setMessage("Processing");
                    builder.show();
                },
                error -> {
                    Toast.makeText(getApplicationContext(), "ERROR: " + error.toString(), Toast.LENGTH_LONG).show();
                }) {
            protected Map<String, String> getParams() {
                Map<String, String> params = new HashMap<>();
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                croppedBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                byte[] byteArray = stream.toByteArray();
                String pictureString = Base64.encodeToString(byteArray, Base64.DEFAULT);
                params.put("image", pictureString);
                return params;
            }
        };
        RequestQueue requestQueue = Volley.newRequestQueue(MainActivity.this);
        requestQueue.add(stringRequest);
        //AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        //builder.setTitle("Sent Successfully");
        //builder.setMessage("Processing");
        Log.d("SendToServer", "Request sent to server");
    }*/

}
