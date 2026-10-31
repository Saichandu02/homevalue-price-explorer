# Data card — California Housing

- **Dataset:** California Housing, block-group aggregates from the 1990 U.S. Census.
- **Origin:** Pace, R. Kelley & Barry, Ronald (1997), "Sparse Spatial Autoregressions",
  Statistics & Probability Letters 33(3), 291–297. Distributed via StatLib / LIBSVM
  and mirrored in the archive used by scikit-learn's `fetch_california_housing`.
- **Download:** `https://ndownloader.figshare.com/files/5976036`
  (archive SHA-256 `aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`,
  MD5 `130d0eececf165046ec4dc621d121d80` — as recorded at acquisition on 2026-10-31).
- **Shipped CSV:** `data/california_housing.csv`, 20,640 rows, 2.0 MB,
  SHA-256 `8e28941c713df5f9835c8f5b0b539cb4e3273967d0fb28070833b5de65409c2a`.
  Regenerate/verify with `scripts/download_data.py`.
- **License / terms:** The dataset is a widely redistributed public research dataset
  (U.S. Census aggregates, no personal data). scikit-learn distributes it for
  research/education use with attribution to Pace & Barry (1997). This repo uses it
  for a portfolio demonstration with that attribution. If you redistribute, keep
  this card. No Vijayawada or other market data is included or claimed.
- **Features (per block group):** Longitude, Latitude, HouseAge (housing median age),
  TotalRooms, TotalBedrooms, Population, Households, MedInc (median income, in
  $10,000s), target MedHouseVal (median house value, USD).
- **Known caveats:** 1990 data — not current prices. Target is capped at $500,001 in
  the source (a visible ceiling effect in residuals). Block-group aggregates, not
  individual properties. Latitude/Longitude let the model learn location effects,
  which dominate in California; error slices by price band/region are reported so
  this is visible rather than hidden.
- **Currency / geography:** USD, California, USA only.
