<?php

if (isset($_POST['title']) && isset($_POST['text'])) {
  $user = $_SERVER['PHP_AUTH_USER'];
  $title = $_POST['title'];
  $text = $_POST['text'];
  $fileName = $user . "_" . $title;
  $fileNameWithExtension = $fileName . '.yaml';
  $input_files = scandir('/etc/sherlockpipe/input');
  $working_files = scandir('/etc/sherlockpipe/working');
  $output_files = scandir('/etc/sherlockpipe/output');
  if (in_array($fileNameWithExtension, $input_files) || in_array($fileNameWithExtension, $working_files) || in_array($fileNameWithExtension, $output_files)) {
      $message = 'There is already a task with the name ' . $fileName;
  } else {
      $file = fopen("/etc/sherlockpipe/input/$fileNameWithExtension", 'w');
      fwrite($file, $text);
      fclose($file);
  }
  // TODO send email acknowledging input file
  header('Location: index.php?message=' . $message);
  exit;
}


?>