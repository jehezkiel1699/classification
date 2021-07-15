<?php 
// $output = shell_exec("D:\\Anaconda\\envs\\skripsi_php\\python.exe tes.py");

include "database.php";

if (isset($_POST['search'])){
	$beritayangdicari = $_POST['textBerita'];
}else{
	$beritayangdicari = $_POST['textBerita'];
}

// $limit = 15;
// $page = isset($_GET['page']) ? $_GET['page'] : 1;
// if(empty($page)){
// 	$start = 0;
// 	$page = 1;
// }else{
// 	$start = ($page - 1) * $limit;
// }

// $start = ($page - 1) * $limit;
// $sql = "SELECT * FROM news WHERE judul LIKE '%$beritayangdicari%' ORDER BY str_to_date(`tanggal`, '%d/%m/%Y') DESC LIMIT $start, $limit";
$sql = "SELECT * FROM news WHERE judul LIKE '%$beritayangdicari%' ORDER BY str_to_date(`tanggal`, '%d/%m/%Y') DESC";
//$sql = "SELECT * FROM news";
$res=mysqli_query($con, $sql);
$news = array();
while ($row = mysqli_fetch_assoc($res)) {
	$news[] = $row;
}

// $sql = "SELECT count(id) AS id FROM news WHERE judul LIKE '%$beritayangdicari%'";
// $res = mysqli_query($con, $sql);
// $newsCount = array();
// while ($row = mysqli_fetch_assoc($res)) {
// 	$newsCount[] = $row;
// }
// $total = $newsCount[0]['id'];
// $pages = ceil($total/$limit);

// $Previous = $page - 1;
// $Next = $page + 1;
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

		<div class="row mt-5 mx-3">
			<div class="col-md-12">
				<form autocomplete="off" action="" method="POST" class="d-flex">
					<input class="form-control me-2" name="textBerita" type="search" placeholder="Search Berita" aria-label="Search" value="<?php echo $beritayangdicari; ?>">
					<!-- <button class="btn btn-outline-success">Search</button> -->
					<input type="submit" name="search" class="btn btn-outline-success" value="Search">
				</form>
			</div>
		</div>

		<div class="row mt-3 mx-3">
			<div class="col-md-12">
				<p class="judul">Hasil Pencarian : <?php echo $beritayangdicari ?></p>
				<hr style="border:1px solid black">
			</div>
		</div>

		<?php foreach ($news as $berita): ?>
		<div class="row mt-3 mx-3">
			<div class="col-md-12">
				<a href="detailberita.php?berita=<?php echo $berita['judul'] ?>" style="text-decoration: none; color:black">
					<div class="card">
						<div class="card-body">
							<div class="row">
								<div class="col-md-3">
									<img src="<?php echo $berita['gambar'] ?>" class="img-thumbnail" style="width:auto; height: 200px">
								</div>
								<div class="col-md-9">
									<p class="judulberita"><?php echo $berita['judul'] ?></p>
									<span class="align-bottom"><?php echo $berita['tanggal'] ?></span>
								</div>
							</div>
						</div>
					</div>
				</a>

				
			</div>
		</div>
		<?php endforeach ?>
		
<!-- 		<div class="row mt-3">
			<div class="col-md-12">
				<nav aria-label="Page navigation example">
					<ul class="pagination justify-content-center"> -->
						<!-- <?php if($start == 0){ ?>
						<li class="page-item">
							<a class="page-link" href="tampilansearch.php?page=<?= $Previous; ?>" tabindex="-1" style="display:none;">Previous</a>
						</li>
						<?php }else{ ?>
						<li class="page-item">
							<a class="page-link" href="tampilansearch.php?page=<?= $Previous; ?>" tabindex="-1">Previous</a>
						</li>
						<?php } ?> -->
						<!--<?php if($page==0){ ?>
						<li class="page-item">
							<a class="page-link" href="index.php?page=<?= $Previous; ?>" tabindex="-1" style="display: none;">Previous</a>
						</li>
						<?php }else{ ?>
							<?php if($page==1){ ?>
								<?php $page=0 ?>
								<a class="page-link" href="index.php" tabindex="-1" >Previous</a>
							<?php }else{ ?>
								<a class="page-link" href="index.php?page=<?= $Previous; ?>" tabindex="-1">Previous</a>
							<?php } ?>
						<?php } ?>-->

						<!-- <?php for($i = 1; $i<= $pages; $i++) :?> -->
							<!-- <li><a class="page-link" href="tampilansearch.php?page=<?= $i; ?>" ><?= $i; ?></a></li> -->
						<!-- <?php endfor; ?> -->

						<!-- <?php if($page != $pages){ ?> -->
						<!-- <li class="page-item">
					    	<a class="page-link" href="tampilansearch.php?page=<?= $Next; ?>">Next</a>
					    </li> -->
					    <!-- <?php } ?> -->
					<!-- </ul>
				</nav>
			
			</div>
		</div> -->



    </div>

    <!-- <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV" crossorigin="anonymous"></script> -->
    <!-- <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.bundle.min.js" integrity="sha384-LtrjvnR4Twt/qOuYxE721u19sVFLVSA4hf/rRt6PrZTmiPltdZcI7q7PXQBYTKyf" crossorigin="anonymous"></script> -->

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/js/bootstrap.bundle.min.js" integrity="sha384-b5kHyXgcpbZJO/tY9Ul7kGkf1S0CWuKcCD38l8YkeH8z8QjE0GmW1gYU5S9FOnJ0" crossorigin="anonymous"></script>
</body>
</html>