import json
import csv
import time
import re
from urllib.parse import urljoin

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError  # type: ignore[import-not-found]
except ModuleNotFoundError:
    sync_playwright = None

    class PlaywrightTimeoutError(Exception):
        pass

    # Install the dependency with: pip install playwright


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://www.tn.gov.in/"
DEPARTMENT_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

OUTPUT_JSON = "schemes.json"
OUTPUT_CSV = "schemes.csv"

# Step 2 Clean Outputs
OUTPUT_CLEAN_JSON = "schemes_cleaned.json"
OUTPUT_CLEAN_CSV = "schemes_cleaned.csv"

DEBUG_DETAIL_HTML = "debug_detail.html"
DEBUG_DETAIL_SCREENSHOT = "debug_detail.png"


# ============================================================
# EXPECTED SCHEME FIELDS
# ============================================================

EXPECTED_FIELDS = [
    "Concerned Department",
    "Concerned District",
    "Organisation Name",
    "Scheme Title/Name",
    "Associated Scheme",
    "Sponsered By",
    "Funding Pattern",
    "Beneficiaries",
    "Types of Benefits",
    "Income",
    "Age From",
    "Age To",
    "Community",
    "How To avail",
    "Introduced On",
    "Description",
    "Scheme Type",
    "Uploaded File",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(text):
    """
    Normalize field labels so small differences in spaces,
    capitalization, colon, slash etc. do not break matching.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove colon
    text = text.replace(":", "")

    # Normalize spaces
    text = " ".join(text.split())

    return text.lower()


def clean_text(text):
    """
    Clean extracted text.
    """

    if text is None:
        return ""

    return " ".join(text.split()).strip()


def get_absolute_url(href):
    """
    Convert relative URL to absolute URL.
    """

    if not href:
        return ""

    return urljoin(BASE_URL, href)


# ============================================================
# EXTRACT FIELD FROM TABLE ROWS
# ============================================================

def extract_from_table(page):
    """
    Extract key/value information from tables.

    The TN scheme detail pages are expected to contain
    table rows where the first cell is the field name
    and the second cell contains the value.
    """

    data = {}

    rows = page.locator("table tr")

    row_count = rows.count()

    print(f"      Table rows found: {row_count}")

    for i in range(row_count):

        row = rows.nth(i)

        try:
            cells = row.locator("th, td")

            cell_count = cells.count()

            if cell_count < 2:
                continue

            label = clean_text(cells.nth(0).inner_text())
            value = clean_text(cells.nth(1).inner_text())

            if not label:
                continue

            if not value:
                continue

            normalized = normalize_label(label)

            data[normalized] = value

        except Exception:
            continue

    return data


# ============================================================
# EXTRACT ALL VISIBLE TEXT / ROW INFORMATION
# ============================================================

def extract_all_rows(page):
    """
    Additional fallback extraction.

    This checks common row structures in case the detail
    page does not use a standard table.
    """

    data = {}

    selectors = [
        "tr",
        ".row",
        ".scheme-row",
        ".scheme-detail-row",
        "li",
    ]

    for selector in selectors:

        try:
            elements = page.locator(selector)

            count = elements.count()

            for i in range(count):

                element = elements.nth(i)

                try:
                    text = clean_text(element.inner_text())

                    if not text:
                        continue

                    # Try splitting common label/value formats
                    for expected in EXPECTED_FIELDS:

                        normalized_expected = normalize_label(expected)

                        normalized_text = normalize_label(text)

                        if normalized_expected in normalized_text:

                            if expected not in data:

                                # Try colon-separated format
                                if ":" in text:

                                    parts = text.split(":", 1)

                                    if len(parts) == 2:

                                        label = clean_text(parts[0])
                                        value = clean_text(parts[1])

                                        if (
                                            normalize_label(label)
                                            == normalized_expected
                                        ):
                                            data[normalized_expected] = value

                except Exception:
                    continue

        except Exception:
            continue

    return data


# ============================================================
# MAP EXTRACTED DATA TO EXPECTED FIELD NAMES
# ============================================================

def map_fields(table_data, fallback_data):
    """
    Convert normalized labels back to the exact field names
    expected by the project.
    """

    result = {}

    for field in EXPECTED_FIELDS:

        normalized_field = normalize_label(field)

        value = ""

        # ----------------------------------------------------
        # Direct table match
        # ----------------------------------------------------

        if normalized_field in table_data:
            value = table_data[normalized_field]

        # ----------------------------------------------------
        # Fallback match
        # ----------------------------------------------------

        elif normalized_field in fallback_data:
            value = fallback_data[normalized_field]

        # ----------------------------------------------------
        # Fuzzy label matching
        # ----------------------------------------------------

        else:

            for key, candidate_value in table_data.items():

                if (
                    normalized_field in key
                    or key in normalized_field
                ):
                    value = candidate_value
                    break

        result[field] = clean_text(value)

    return result


# ============================================================
# SCRAPE ONE DETAIL PAGE
# ============================================================

def scrape_detail_page(page, scheme_name, detail_url, department, debug=False):

    print("\n--------------------------------------------------")
    print(f"Opening scheme: {scheme_name}")
    print(f"URL: {detail_url}")
    print("--------------------------------------------------")

    try:

        page.goto(
            detail_url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        # Give the page a short time to finish rendering
        page.wait_for_timeout(1500)

    except PlaywrightTimeoutError:

        print("      WARNING: Page load timeout")

        # Continue because the page may still contain data

    except Exception as e:

        print(f"      ERROR loading page: {e}")

        return None

    # --------------------------------------------------------
    # Save debug page for FIRST DETAIL PAGE
    # --------------------------------------------------------

    if debug:

        try:

            with open(
                DEBUG_DETAIL_HTML,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(page.content())

            page.screenshot(
                path=DEBUG_DETAIL_SCREENSHOT,
                full_page=True
            )

            print(f"      Saved {DEBUG_DETAIL_HTML}")
            print(f"      Saved {DEBUG_DETAIL_SCREENSHOT}")

        except Exception as e:

            print(f"      Debug save error: {e}")

    # --------------------------------------------------------
    # Extract table data
    # --------------------------------------------------------

    table_data = extract_from_table(page)

    # --------------------------------------------------------
    # Extract fallback data
    # --------------------------------------------------------

    fallback_data = extract_all_rows(page)

    # --------------------------------------------------------
    # Map to expected fields
    # --------------------------------------------------------

    fields = map_fields(
        table_data,
        fallback_data
    )

    # --------------------------------------------------------
    # Ensure basic information exists
    # --------------------------------------------------------

    if not fields["Concerned Department"]:
        fields["Concerned Department"] = department

    if not fields["Scheme Title/Name"]:
        fields["Scheme Title/Name"] = scheme_name

    # --------------------------------------------------------
    # Add metadata useful for KG-RAG
    # --------------------------------------------------------

    record = {
        "id": detail_url.split("id=")[-1]
        if "id=" in detail_url else "",

        "department": department,

        "scheme_name": scheme_name,

        "detail_url": detail_url,

        "fields": fields,
    }

    # --------------------------------------------------------
    # Show extraction result
    # --------------------------------------------------------

    filled = 0

    for field, value in fields.items():

        if value:

            filled += 1

            print(f"      {field}: {value}")

    print(
        f"      Fields populated: {filled}/{len(EXPECTED_FIELDS)}"
    )

    return record


# ============================================================
# SCRAPE MAIN DEPARTMENT PAGE
# ============================================================

def scrape():

    records = []

    with sync_playwright() as p:

        # ----------------------------------------------------
        # Launch browser
        # ----------------------------------------------------

        browser = p.chromium.launch(
            headless=False
        )

        context = browser.new_context(
            viewport={
                "width": 1440,
                "height": 900
            }
        )

        page = context.new_page()

        # ----------------------------------------------------
        # Open department page
        # ----------------------------------------------------

        print("=" * 70)
        print("TN GOVERNMENT SCHEME SCRAPER")
        print("=" * 70)

        print(f"\nLoading department page:")
        print(DEPARTMENT_URL)

        try:

            page.goto(
                DEPARTMENT_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(2000)

        except PlaywrightTimeoutError:

            print("\nWARNING: Department page timed out.")
            print("Checking whether content was loaded...")

        except Exception as e:

            print(f"\nERROR loading department page:")
            print(e)

            browser.close()

            return records

        # ----------------------------------------------------
        # Extract department
        # ----------------------------------------------------

        department = ""

        try:

            department = clean_text(
                page.locator("#content1").inner_text()
            )

        except Exception:

            print(
                "WARNING: #content1 not found."
            )

        print("\nDepartment:")
        print(department)

        # ----------------------------------------------------
        # Find scheme links
        # ----------------------------------------------------

        print("\nSearching for scheme links...")

        links = page.locator(
            "#content a[href*='scheme_details.php']"
        )

        try:

            count = links.count()

        except Exception:

            count = 0

        print(f"\nFound {count} scheme links")

        if count == 0:

            print(
                "\nERROR: No scheme links found."
            )

            print(
                "Check debug_page.html and debug_screenshot.png"
            )

            browser.close()

            return records

        # ----------------------------------------------------
        # Collect scheme links first
        # ----------------------------------------------------

        scheme_links = []

        for i in range(count):

            try:

                link = links.nth(i)

                scheme_name = clean_text(
                    link.inner_text()
                )

                href = link.get_attribute("href")

                detail_url = get_absolute_url(href)

                if not scheme_name:
                    continue

                if not detail_url:
                    continue

                scheme_links.append(
                    {
                        "scheme_name": scheme_name,
                        "detail_url": detail_url
                    }
                )

                print(
                    f"{i + 1}. {scheme_name}"
                )

                print(
                    f"   {detail_url}"
                )

            except Exception as e:

                print(
                    f"Error reading link {i + 1}: {e}"
                )

        print(
            f"\nTotal valid scheme links: {len(scheme_links)}"
        )

        # ----------------------------------------------------
        # Create detail page
        # ----------------------------------------------------

        detail_page = context.new_page()

        # ----------------------------------------------------
        # Visit every scheme
        # ----------------------------------------------------

        total = len(scheme_links)

        for index, scheme in enumerate(
            scheme_links,
            start=1
        ):

            print("\n")
            print("=" * 70)
            print(
                f"PROCESSING {index}/{total}"
            )
            print("=" * 70)

            try:

                record = scrape_detail_page(
                    page=detail_page,

                    scheme_name=scheme[
                        "scheme_name"
                    ],

                    detail_url=scheme[
                        "detail_url"
                    ],

                    department=department,

                    # Save first detail page for DOM inspection
                    debug=(index == 1)
                )

                if record:

                    records.append(record)

            except Exception as e:

                print(
                    f"ERROR processing scheme: {e}"
                )

            # Small delay to avoid hammering the website
            time.sleep(0.5)

        # ----------------------------------------------------
        # Close browser
        # ----------------------------------------------------

        detail_page.close()
        page.close()

        browser.close()

    return records


# ============================================================
# SAVE JSON
# ============================================================

def save_json(records):

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nSaved JSON: {OUTPUT_JSON}"
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    if not records:

        print(
            "No records available for CSV."
        )

        return

    fieldnames = [
        "id",
        "department",
        "scheme_name",
        "detail_url"
    ] + EXPECTED_FIELDS

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:

            row = {
                "id": record.get("id", ""),
                "department": record.get(
                    "department", ""
                ),
                "scheme_name": record.get(
                    "scheme_name", ""
                ),
                "detail_url": record.get(
                    "detail_url", ""
                )
            }

            fields = record.get(
                "fields",
                {}
            )

            for field in EXPECTED_FIELDS:

                row[field] = fields.get(
                    field,
                    ""
                )

            writer.writerow(row)

    print(
        f"Saved CSV: {OUTPUT_CSV}"
    )


# ============================================================
# STEP 2: DATA VALIDATION & CLEANING (ADDED AT THE END)
# ============================================================

NULL_EQUIVALENTS = {
    "", "n/a", "na", "nil", "none", "null", "-", "--", "not applicable", "no"
}

def clean_field_value(val):
    """Sanitize strings, handle spaces, remove null-like values."""
    if val is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(val)).replace("\xa0", " ").strip()
    return None if cleaned.lower() in NULL_EQUIVALENTS else cleaned

def parse_int_value(val):
    """Extract integer value from a string (e.g. '18 Years' -> 18)."""
    text = clean_field_value(val)
    if not text:
        return None
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None

def split_field_list(val):
    """Split comma/slash/semicolon separated entities into arrays for Neo4j."""
    text = clean_field_value(val)
    if not text:
        return []
    tokens = re.split(r"[,;/]+", text)
    cleaned_tokens = []
    for t in tokens:
        c = clean_field_value(t)
        if c and len(c) > 1:
            cleaned_tokens.append(c)
    return list(dict.fromkeys(cleaned_tokens))

def parse_income_limit(val):
    """Extract numeric income threshold."""
    text = clean_field_value(val)
    if not text:
        return None
    cleaned = text.replace(",", "").replace("/-", "")
    match = re.search(r"(?:rs\.?|inr)?\s*(\d{4,9})", cleaned, re.IGNORECASE)
    return int(match.group(1)) if match else None

def validate_and_clean_data(records):
    """
    Step 2 Processing:
    Validates data, structures lists, extracts bounds,
    and prepares composite text for FAISS embedding.
    """
    print("\n" + "=" * 70)
    print("STEP 2: VALIDATING & CLEANING DATA")
    print("=" * 70)

    cleaned_records = []
    seen_ids = set()

    for r in records:
        fields = r.get("fields", {})
        scheme_name = clean_field_value(r.get("scheme_name")) or clean_field_value(fields.get("Scheme Title/Name"))
        department = clean_field_value(r.get("department")) or clean_field_value(fields.get("Concerned Department"))

        if not scheme_name:
            continue

        raw_income = clean_field_value(fields.get("Income"))
        age_min = parse_int_value(fields.get("Age From"))
        age_max = parse_int_value(fields.get("Age To"))

        if age_min and age_max and age_min > age_max:
            age_min, age_max = age_max, age_min

        beneficiaries = split_field_list(fields.get("Beneficiaries"))
        benefits = split_field_list(fields.get("Types of Benefits"))
        communities = split_field_list(fields.get("Community"))
        desc = clean_field_value(fields.get("Description"))
        avail = clean_field_value(fields.get("How To avail"))

        # Composite text constructed for Step 3 FAISS Embeddings
        composite_parts = [
            f"Scheme: {scheme_name}",
            f"Department: {department or 'N/A'}",
            f"Description: {desc or 'N/A'}",
            f"Target Beneficiaries: {', '.join(beneficiaries) if beneficiaries else 'All'}",
            f"Target Communities: {', '.join(communities) if communities else 'All'}",
            f"Benefits: {', '.join(benefits) if benefits else 'General Support'}",
            f"How to Apply: {avail or 'Refer to department'}",
        ]
        if age_min or age_max:
            composite_parts.append(f"Age Limit: {age_min or 0} to {age_max or 'No limit'}")
        if raw_income:
            composite_parts.append(f"Income Limit: {raw_income}")

        composite_text = "\n".join(composite_parts)
        scheme_id = str(r.get("id") or "")

        # Deduplication
        dedup_key = scheme_id if scheme_id else scheme_name.lower()
        if dedup_key in seen_ids:
            continue
        seen_ids.add(dedup_key)

        cleaned_item = {
            "id": scheme_id,
            "scheme_name": scheme_name,
            "department": department or "Unassigned",
            "detail_url": clean_field_value(r.get("detail_url")) or "",
            "concerned_district": clean_field_value(fields.get("Concerned District")),
            "organisation_name": clean_field_value(fields.get("Organisation Name")),
            "associated_scheme": clean_field_value(fields.get("Associated Scheme")),
            "sponsored_by": clean_field_value(fields.get("Sponsered By")),
            "funding_pattern": clean_field_value(fields.get("Funding Pattern")),
            "scheme_type": clean_field_value(fields.get("Scheme Type")),
            "introduced_on": clean_field_value(fields.get("Introduced On")),
            "uploaded_file": clean_field_value(fields.get("Uploaded File")),
            "description": desc,
            "how_to_avail": avail,
            "raw_income_criteria": raw_income,
            "income_max_inr": parse_income_limit(raw_income),
            "age_min": age_min,
            "age_max": age_max,
            "beneficiaries": beneficiaries,
            "communities": communities,
            "types_of_benefits": benefits,
            "composite_text": composite_text
        }
        cleaned_records.append(cleaned_item)

    # 1. Save schemes_cleaned.json
    with open(OUTPUT_CLEAN_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

    # 2. Save schemes_cleaned.csv
    if cleaned_records:
        with open(OUTPUT_CLEAN_CSV, "w", newline="", encoding="utf-8-sig") as f:
            headers = [k for k in cleaned_records[0].keys() if k != "composite_text"]
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for obj in cleaned_records:
                row = dict(obj)
                row.pop("composite_text", None)
                for k, v in row.items():
                    if isinstance(v, list):
                        row[k] = ", ".join(v)
                writer.writerow(row)

    print(f"STEP 2 FINISHED:")
    print(f"  Valid & cleaned schemes: {len(cleaned_records)} / {len(records)}")
    print(f"  Output saved: {OUTPUT_CLEAN_JSON}")
    print(f"  Output saved: {OUTPUT_CLEAN_CSV}")
    return cleaned_records


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    start_time = time.time()

    # Step 1: Run your existing scraping logic
    records = scrape()

    print("\n")
    print("=" * 70)
    print("SCRAPING COMPLETED")
    print("=" * 70)

    print(
        f"\nTOTAL SCHEMES SCRAPED: {len(records)}"
    )

    # Save Step 1 raw outputs (Original)
    save_json(records)
    save_csv(records)

    # Step 2: Validate and clean the data
    clean_records = validate_and_clean_data(records)

    elapsed = time.time() - start_time

    print(
        f"\nTotal time taken: {elapsed:.2f} seconds"
    )

    print("\nFiles generated:")
    print(f"  Raw:   {OUTPUT_JSON}, {OUTPUT_CSV}")
    print(f"  Clean: {OUTPUT_CLEAN_JSON}, {OUTPUT_CLEAN_CSV}")
    print("\nDone.")