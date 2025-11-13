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
            <img src="images/Molecule_simulation.jpg" alt="Project Screenshot">
            <div class="text-block">
                <h2>About the Project</h2>
                <p>
                    In this project, I worked in a team to create a model that makes connections between molecules where charges are likely to "jump" between. <br>
                    The main challenge of this project was trying to figure out which atoms can form a connection. A connection can be form if and only if:<br>
                    <ol>
                        <li>No other molecule physically blocks a connection</li>
                        <li>The molecule is not blocked by another part of itself from making a connection to another molecule </li>
                    </ol>
                </p>
            </div>
        </section>

        <section class="image-section">
            <div class="image-desc-item">
            <img src="images/Molecule.jpg" alt="Molecule">
            <p>This is how the molecule that we are using in this particular setup looks like.</p>
            </div>
            <div class="image-desc-item">
            <img src="images/Connection_condition.jpg" alt="Connection_condition">
            <p>These are the cases where connections can't be made</p>
            </div>
        </section>

        <section class="image-section">
            <div class="function-call-png">
            <img src="images/function-call.png" alt="Function call diagram">
            <p>Function call diagram(Flow Chart)</p>
            </div>
        </section>

        <script src="prism.js"></script>
        <section class="code-section">
            <h2>Code Highlights</h2>
            <p><a href="https://github.com/QMathsdude/kmc-charge-transport-organic-semiconductor">Github Link</a>

            <h3>/utils/blocking_algo.py</h3>
            <pre><code class="language-python">
            <?
            echo htmlspecialchars(file_get_contents("code/blocking_algo.py"))
            ?>
            </code></pre>

            <h3>/utils/networkx_graph_making.py</h3>
            <pre><code class="language-python">
            <?
            echo htmlspecialchars(file_get_contents("code/networkx_graph_making.py"))
            ?>
            </code></pre>

            <h3>/utils/subbox_division.py</h3>
            <pre><code class="language-python">
            <?
            echo htmlspecialchars(file_get_contents("code/subbox_division.py"))
            ?>
            </code></pre>
        </section>
    </main>

    <!-- Footer -->
    <?php include 'includes/footer.php' ?>
</body>
</html>
