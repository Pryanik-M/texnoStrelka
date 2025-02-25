let currentIndex = 0;

function showSlide(index) {
  const carouselInner = document.querySelector('.carousel-inner');
  const itemWidth = document.querySelector('.carousel-item').offsetWidth;
  const offset = -index * itemWidth; // Сдвигаем на один элемент
  carouselInner.style.transform = `translateX(${offset}px)`;
}

function nextSlide() {
  const totalItems = document.querySelectorAll('.carousel-item').length;
  if (currentIndex < totalItems - 4) { // Ограничиваем сдвиг, чтобы не выйти за пределы
    currentIndex += 1;
    showSlide(currentIndex);
    console.log("Работает");
  }
}

function prevSlide() {
  if (currentIndex > 0) { // Ограничиваем сдвиг, чтобы не выйти за пределы
    currentIndex -= 1;
    showSlide(currentIndex);
  }
}

// Инициализация первой видимой группы элементов
showSlide(currentIndex);
