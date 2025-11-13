<?php $page_title = "Molecule Simulation"; ?>    
<!DOCTYPE html>
<html lang="en">

<!-- HEAD -->
<?php include 'includes/head.php'; ?>

<body>
     <!-- HEADER -->
    <?php include 'includes/header.php' ?>

    <main>
        <section class="image-text-section">
            <img src="images/GenQ.jpg" alt="Project Screenshot">
            <div class="text-block">
                <h2>About the Project</h2>
                <p>
                    In this, my team and I attended a hackathon where we were tasked with doing portfolio optimisation using quantum computing algorithms. <br>
                    The main challenge of this project was:<br>
                    <ol>
                        <li>Creating a quantum computing model from scratch</li>
                        <li>Providing innovations</li>
                        <li>Proving quantum advantage</li>
                        <li>Pitching</li>
                        <li>Doing all this in less than 36 hours</li>
                    </ol>
                </p>
            </div>
        </section>
        
        <section class="image-section">
            <div class="function-call-png">
            <video src="images/GenQ_Demo.mp4" controls alt="Demo Video"></video>
            <p>Product Demo Video</p>
            </div>
        </section>

        <script src="prism.js"></script>
        <section class="code-section">
            <h2>Code Highlights</h2>
            <p><a href="https://github.com/LeCx128/QAI_hackathon#">Github Link</a>

            <h3>/utils/Json_parse.py</h3>
            <pre><code class="language-python">
            <?
            echo htmlspecialchars(file_get_contents("code/Json_parse.py"))
            ?>
            </code></pre>

            <h3>/utils/QAOA_portfolio_optimize.py</h3>
            <pre><code class="language-python">
            <?
            echo htmlspecialchars(file_get_contents("code/QAOA_portfolio_optimize.py"))
            ?>
            </code></pre>
        </section>

    </main>

    <!-- Footer -->
    <?php include 'includes/footer.php' ?>
</body>
</html>
