<?php
// define variables and set to empty values
$partno = "";

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
        echo "<br>";
        $partno = $_POST["partno"];
        $inputData = $partno;
        
        echo "<h3>Forecast Result of " . " $partno </h3>";
        exec("python python\alpha\get-forecast.py  \"$inputData\"", $output);
        if ($output == true) {
            // ########################################################################################
            // Load the JSON file
            $data = json_decode(file_get_contents('./python/alpha/output/result.json'), true);

            // Display general information
            echo "P/N: " . $data['pn'] . "<br>";
            echo "Best Model: " . $data['best_model'] . "<br>";
            echo "Best FD: " . $data['best_fd'] . "<br>";

            /* display image */
            $allImagePath = "/python/alpha/output/plotsemua.png";
            $bestImagePath = "/python/alpha/output/plotbest.png";
            echo "<h3>All Plot  |  Best Plot</h3>";
            echo "<img src='" . $allImagePath . "' alt='Plot Image' style='width:50%; height:auto;'>";
            echo "<img src='" . $bestImagePath . "' alt='Plot Image' style='width:50%; height:auto;'>";

            // Display metrics
            echo "<h3>Metrics:</h3>";
            echo "<table border='1' cellpadding='5'>";
            echo "<tr><th>Model</th><th>RMSE</th><th>R2</th></tr>";
            foreach ($data['metric'] as $metric) {
                echo "<tr>";
                echo "<td>" . $metric['Model'] . "</td>";
                echo "<td>" . $metric['RMSE'] . "</td>";
                echo "<td>" . $metric['R2'] . "</td>";
                echo "</tr>";
            }
            echo "</table>";

            // Display data for each period
            echo "<h3>Data:</h3>";
            echo "<table border='1' cellpadding='5'>";
            echo "
                <tr>
                    <th>Period</th>
                    <th>Quantity</th>
                    <th>MA Prediction</th>
                    <th>Smoothed Weighted Prediction</th>
                    <th>Linear Regression Prediction</th>
                    <th>Polynomial Regression Prediction (Degree 2)</th>
                    <th>Polynomial Regression Prediction (Degree 3)</th>
                    <th>SES Prediction</th>
                    <th>DES Prediction</th>
                </tr>
            ";
            foreach ($data['data'] as $entry) {
                echo "<tr>";
                echo "<td>" . $entry['Period'] . "</td>";
                echo "<td>" . $entry['Quantity'] . "</td>";
                echo "<td>" . $entry['MA_Prediction'] . "</td>";
                echo "<td>" . $entry['Smoothed_Weighted_Prediction'] . "</td>";
                echo "<td>" . $entry['Linear_Regression_Prediction'] . "</td>";
                echo "<td>" . $entry['Polynomial_Regression_Prediction_Degree_2'] . "</td>";
                echo "<td>" . $entry['Polynomial_Regression_Prediction_Degree_3'] . "</td>";
                echo "<td>" . $entry['SES_Prediction'] . "</td>";
                echo "<td>" . $entry['DES_Prediction'] . "</td>";
                echo "</tr>";
            }
            echo "</table>";

                     


        } // end of if python output true
    }
    ?>





</body>

</html>