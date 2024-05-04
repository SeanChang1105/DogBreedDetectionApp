<?php
  $pyexe = '/data/lam138/myvenv/bin/python3 /data/lam138/Model/modelPredict.py';

  $descriptorspec = array(
    0 => array("pipe", "r"),  // stdin is a pipe that the child will read from
    1 => array("pipe", "w"),  // stdout is a pipe that the child will write to
    2 => array("pipe", "w")   // stderr is a pipe that the child will write to
  );

  $process = proc_open($pyexe, $descriptorspec, $pipes);

  if (is_resource($process)) {
      $stdout = stream_get_contents($pipes[1]);
      fclose($pipes[1]);

      $stderr = stream_get_contents($pipes[2]);
      fclose($pipes[2]);

      $status = proc_get_status($process);
      proc_close($process);

      if ($status['exitcode'] === 0) {
          # If everything went smoothly this is the output
          echo $stdout;
      } else {
          echo "Error:\n";
          echo $stderr;
      }
  } else {
      echo "Failed to execute Python script.";
  }
?>