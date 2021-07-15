<?php 
// $output = shell_exec("D:\\Anaconda\\envs\\skripsi_php\\python.exe tes.py");
include "database.php";
if (isset($_GET['berita'])){
	$judulBerita = $_GET['berita'];
	// print($judulBerita);
	$sql = "SELECT * FROM news WHERE judul='$judulBerita'";
	$res = mysqli_query($con, $sql);
	$berita = [];
	#while($row =mysqli_fetch_assoc($res)){ $judul[] = $row['judul'];$isiberita[] = $row['isi'];}
	while ($row = mysqli_fetch_assoc($res)) {
		$berita = $row;
	}
}else{

}
 ?>

<!DOCTYPE html>
<html>
<head>
	<!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <!-- Bootstrap CSS -->
    <!-- <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous"> -->

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-BmbxuPwQa2lc/FVzBcNJ7UAyJxM6wuqIj61tLrc4wSX0szH/Ev+nYRRuWlolflfl" crossorigin="anonymous">
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:wght@600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="styles.css">
    <title>Portal Berita</title>
</head>
<body>
	
	<div class="container-fluid mt-2 ">
		<!-- Navbar -->
		<div class="row">
			<div class="col-md-12">
				<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
					<div class="container-fluid">
						<a class="navbar-brand" href="index.php">Portal Berita</a>
						<button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
							<span class="navbar-toggler-icon"></span>
						</button>
						<div class="collapse navbar-collapse" id="navbarSupportedContent">
							<ul class="navbar-nav me-auto mb-2 mb-lg-0">
								<!-- health, edukasi, teknologi, lifestyle dan sport. -->
								<li class="nav-item">
									<a class="nav-link active" aria-current="page" href="health.php">Health</a>
								</li>
								<li class="nav-item">
									<a class="nav-link active" href="edukasi.php">Edukasi</a>
								</li>
								<li class="nav-item">
									<a class="nav-link active" href="teknologi.php">Teknologi</a>
								</li>
								<li class="nav-item">
									<a class="nav-link active" href="lifestyle.php">LifeStyle</a>
								</li>
								<li class="nav-item">
									<a class="nav-link active" href="sport.php">Sport</a>
								</li>
								<!-- <li class="nav-item">
									<a class="nav-link active" href="indeksberita.php">Indeks</a>
								</li> -->
							</ul>
							<!-- <form class="d-flex">
								<input class="form-control me-2" type="search" placeholder="Search" aria-label="Search">
								<button class="btn btn-outline-success" type="submit">Search</button>
							</form> -->
						</div>
					</div>
				</nav>
			</div>
		</div>

		<div class="row mt-3 mx-3">
			<div class="col-md-12">
				<div class="row">
					<div class="col-md-12">
						<!-- Headline -->
						<div class="row">
							<div class="col-md-12"><p class="judulberita"><?php echo $berita['judul'] ?></p></div>
						</div>
						<div class="row">
							<div class="col-md-12"><p><?php echo $berita['tanggal'] ?></p></div>
						</div>

					</div>
				</div>
			</div>
		</div>

		<div class="row mt-3 mx-3">
			<div class="col-md-12">
				<p class="isiberita"><?php echo $berita['isi'] ?></p>
			</div>
			
		</div>
        
        <div class="row mt-3 mx-3">
        	<!-- <?php $tagBerita = str_replace("<br>", ",", $berita['prediksi']); ?> -->
        	<p>Tags :</p> 
        	<?php $kategori= explode("<br>", $berita['prediksi']); ?>
        	<?php foreach ($kategori as $cat ) { ?>
        		<?php $urlToKategori = explode(" ", $cat); ?>
        		<a href="tagsearch.php?tag=<?php echo $urlToKategori[0]; ?>"><?php echo $cat; ?></a>
        	<?php } ?>
        	
        </div>

		      
    </div>

    <!-- <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV" crossorigin="anonymous"></script> -->
    <!-- <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.bundle.min.js" integrity="sha384-LtrjvnR4Twt/qOuYxE721u19sVFLVSA4hf/rRt6PrZTmiPltdZcI7q7PXQBYTKyf" crossorigin="anonymous"></script> -->

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/js/bootstrap.bundle.min.js" integrity="sha384-b5kHyXgcpbZJO/tY9Ul7kGkf1S0CWuKcCD38l8YkeH8z8QjE0GmW1gYU5S9FOnJ0" crossorigin="anonymous"></script>
</body>
</html>