package com.example.dogidentification;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.canhub.cropper.CropImageView;

public class Processing extends AppCompatActivity {
    int if_gallery = MainActivity.if_gallery;

    Button uploadButton;
    String serverUrl = "https://bluepill.ecn.purdue.edu/~lam138/uploadImage.php";
    String executeUrl = "https://bluepill.ecn.purdue.edu/~lam138/executePy.php";

    Bitmap image;
    Button exitButton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_processing);
        if (if_gallery == 1) {
            Uri imageUri = Uri.parse(getIntent().getStringExtra("imageUri"));
            try {
                image = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        } else {
            image = getIntent().getParcelableExtra("image");
        }
        CropImageView cropImageView = findViewById(R.id.cropImageView);
        cropImageView.setImageBitmap(image);
        cropImageView.setOnCropImageCompleteListener((view, result) -> {
            Bitmap cropped = result.getBitmap();
        });
        uploadButton = findViewById(R.id.upload_button);
        uploadButton.setOnClickListener(v -> {
            Bitmap cropped = cropImageView.getCroppedImage();
            sendImgToServer(cropped);
            retrieveResult();
        });

        exitButton = findViewById(R.id.main_button);
        exitButton.setOnClickListener(v -> {
            navigateToMainActivity();
        });

    }

    private void navigateToMainActivity() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        startActivity(intent);
        finish();
    }

    private void sendImgToServer(Bitmap croppedBitmap) {
        StringRequest stringRequest = new StringRequest(Request.Method.POST, serverUrl,
                response -> {
                    Toast.makeText(getApplicationContext(), response, Toast.LENGTH_LONG).show();
                    AlertDialog.Builder builder = new AlertDialog.Builder(Processing.this);
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
        RequestQueue requestQueue = Volley.newRequestQueue(Processing.this);
        requestQueue.add(stringRequest);
        //AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        //builder.setTitle("Sent Successfully");
        //builder.setMessage("Processing");
        Log.d("SendToServer", "Request sent to server");
    }

    private void retrieveResult() {
        StringRequest stringRequest = new StringRequest(Request.Method.GET, executeUrl,
                response -> {
                    Toast.makeText(getApplicationContext(), response, Toast.LENGTH_LONG).show();
                    AlertDialog.Builder builder = new AlertDialog.Builder(Processing.this);
                    builder.setTitle("Result");
                    builder.setMessage(response);
                    builder.show();
                    },
                error -> Toast.makeText(getApplicationContext(), "ERROR: " + error.toString(), Toast.LENGTH_LONG).show());

        stringRequest.setRetryPolicy(new DefaultRetryPolicy(
                30000,  // 30 seconds
                DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));

        // Add request to request queue
        RequestQueue requestQueue = Volley.newRequestQueue(Processing.this);
        requestQueue.add(stringRequest);
    }


}
