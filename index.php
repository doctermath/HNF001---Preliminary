<?php
// define variables and set to empty values
$partno = "129888";

?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHNF01</title>
</head>

<body>
    <h2>
        Parts stock prediction using various mathematical forecasting methods
    </h2>

    <!-- hello -->
    <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
        Part No: <input type="text" name="partno" value="<?php echo $partno; ?>">
        
        <br><br>
        <input type="submit" name="submit" value="Submit">
    </form>

    <?php
    if ($_SERVER["REQUEST_METHOD"] == "POST" && !(empty($_POST["partno"]))) {
        echo "<br><br>";
        $partno = $_POST["partno"];
          
        echo "<h3>Forecast Result of " . " $partno </h3>";
        exec("python python\alpha\get-forecast.py", $output);
        if ($output == true) {
            echo "makanmalam";
        }
          
        
    }
    ?>


    


</body>

</html>