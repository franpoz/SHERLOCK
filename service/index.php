<?php

if (isset($_GET['message'])) {
    echo $_GET['message'] . '<br>';
}
echo '<form action="form.php" method="post">';
echo '<label for="textfield">Task ID:</label><br>';
echo '<input type="text" id="title" name="title"><br>';
echo '<label for="textarea">Task YAML definition:</label><br>';
echo '<textarea id="text" name="text"></textarea><br>';
echo '<input type="submit" value="Submit">';
echo '</form>';

$files = scandir('/etc/sherlockpipe/input');
echo 'There are ' . (count($files) - 2) . ' pending tasks with names:<br>';
foreach ($files as $file) {
    if ($file != '.' && $file != '..') {
        echo $file . '<br>';
    }
}

?>