// Оценка видео
const stars = document.querySelectorAll('.star');
const selectedRating = document.getElementById('selected-rating');

stars.forEach((star, index) => {
    star.addEventListener('click', function() {
        const value = parseInt(this.getAttribute('data-value'));
        selectedRating.textContent = value;

        // Подсветка выбранных звезд
        stars.forEach((s, idx) => {
            s.classList.toggle('active', idx < value);
        });
    });

    // Опционально: Добавляем hover-эффект
    star.addEventListener('mouseover', () => {
        const hoverValue = parseInt(star.getAttribute('data-value'));
        stars.forEach((s, idx) => {
            s.classList.toggle('hover', idx < hoverValue);
        });
    });

    star.addEventListener('mouseout', () => {
        stars.forEach(s => s.classList.remove('hover'));
    });
});