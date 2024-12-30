<?php
// Execute Python File 
exec("python python/process-allparts-forecast.py", $output);
if ($output == true) {
    // Read the JSON data from file
    $jsonData = file_get_contents('output/result.json');
    $data = json_decode($jsonData, true);
} else {
    die("Python Script Failed");
}

// Function to pretty print arrays and nested data
function prettyPrint($data)
{
    echo '<pre>' . print_r($data, true) . '</pre>';
}
?>


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BG PROCESS</title>
</head>
<body>
    <h1>data</h1>
    <!-- Process Neccesarry Data -->
    <?php
    foreach($Data) {
        $NewData = 
        
    }



    ?>

    <div>
        <?php prettyPrint($NewData) ?>
    </div>
</body>
</html>