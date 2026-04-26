import os
import sys
import json
import urllib.request
import urllib.parse
import pandas as pd
from datetime import datetime, timezone
import isodate
import time
import random
import string

# ── YouTube Data API v3 Keys ─────────────────────────────────────────────────
# Set your API keys as a comma-separated environment variable:
#   export YOUTUBE_API_KEYS="key1,key2,key3"       (Linux/macOS)
#   $env:YOUTUBE_API_KEYS = "key1,key2,key3"       (PowerShell)
#
# Obtain keys from: https://console.cloud.google.com/apis/credentials
# Enable "YouTube Data API v3" for each project.
# ─────────────────────────────────────────────────────────────────────────────

API_KEYS = os.environ.get("YOUTUBE_API_KEYS", "").split(",")
if not API_KEYS or API_KEYS == [""]:
    raise ValueError(
        "No API keys found. Set the YOUTUBE_API_KEYS environment variable.\n"
        "Example: export YOUTUBE_API_KEYS='AIzaSy...,AIzaSy...,AIzaSy...'"
    )
current_key_idx = 0

def get_api_key():
    global current_key_idx
    return API_KEYS[current_key_idx]

def rotate_api_key():
    global current_key_idx
    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
    print(f"Rotated to next API key (index {current_key_idx})", flush=True)

def fetch_json(url_template):
    for _ in range(len(API_KEYS)):
        url = url_template.format(key=get_api_key())
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in [403, 429]: 
                print(f"HTTP {e.code} error. Quota exceeded or forbidden. Rotating...", flush=True)
                rotate_api_key()
            else:
                return None
        except Exception as e:
            return None
    print("All API keys failed or exhausted.")
    return "STOP"

def append_to_csv_safe(df, csv_file, header):
    """Tries to append to CSV. If locked, tries alternative filenames."""
    filename = csv_file
    for attempt in range(10):
        try:
            df.to_csv(filename, mode='a', header=header, index=False, encoding='utf-8-sig')
            return filename
        except PermissionError:
            print(f"Warning: {filename} is locked. Trying a fallback...", flush=True)
            filename = f"youtube_data_fallback_{attempt}.csv"
            header = not os.path.exists(filename)
            time.sleep(1)
    
    print("Could not write CSV. All fallback names locked.", flush=True)
    return csv_file

def collect_data():
    csv_file = "youtube_data.csv"
    existing_video_ids = set()
    
    base_languages_queries = {
        "en": ["vlog", "tech", "gaming", "music", "news", "education", "comedy", "science", "sports", "howto"],
        "hi": ["व्लॉग", "टेक", "गेमिंग", "संगीत", "समाचार", "शिक्षा", "कॉमेडी", "विज्ञान", "खेल", "फिल्में"],
        "bn": ["ব্লগ", "টেক", "গেমিং", "গান", "খবর", "শিক্ষা", "কমেডি", "বিজ্ঞান", "খেলা", "সিনেমা"],
        "te": ["వ్లాగ్", "టెక్", "గేమింగ్", "సంగీతం", "వార్తలు", "విద్య", "కామెడీ", "సైన్స్", "క్రీడలు", "సినిమాలు"],
        "mr": ["व्लॉग", "तंत्रज्ञान", "गेमिंग", "संगीत", "बातम्या", "शिक्षण", "विनोद", "विज्ञान", "खेळ", "चित्रपट"],
        "ta": ["வ்லாக்", "தொழில்நுட்பம்", "கேமிங்", "இசை", "செய்திகள்", "கல்வி", "நகைச்சுவை", "அறிவியல்", "விளையாட்டு", "சினிமா"],
        "gu": ["વ્લોગ", "ટેકનોલોજી", "ગેમિંગ", "સંગીત", "સમાચાર", "શિક્ષણ", "કોમેડી", "વિજ્ઞાન", "રમતગમત", "ફિલ્મો"],
        "ur": ["ولاگ", "ٹیک", "گیمنگ", "موسیقی", "خبریں", "تعلیم", "کامیڈی", "سائنس", "کھیل", "فلمیں"],
        "kn": ["ವ್ಲಾಗ್", "ಟೆಕ್", "ಗೇಮಿಂಗ್", "ಸಂಗೀತ", "ಸುದ್ದಿ", "ಶಿಕ್ಷಣ", "ಕಾಮಿಡಿ", "ವಿಜ್ಞಾನ", "ಆಟಗಳು", "ಚಲನಚಿತ್ರಗಳು"],
        "es": ["vlog", "tecnología", "juegos", "música", "noticias", "educación", "comedia", "ciencia", "deportes", "películas"],
        "fr": ["vlog", "technologie", "jeux vidéo", "musique", "actualités", "éducation", "comédie", "science", "sports", "cinéma"],
        "ko": ["브이로그", "기술", "게임", "음악", "뉴스", "교육", "코미디", "과학", "스포츠", "영화"],
        "de": ["vlog", "technologie", "gaming", "musik", "nachrichten", "bildung", "komödie", "wissenschaft", "sport", "filme"]
    }
    
    # Generate completely randomized, unbiased alphanumeric queries
    all_possible_queries = list(string.ascii_lowercase) # 'a' to 'z'
    
    # Add all 2-letter combinations (aa, ab, ac ... zz)
    for b1 in string.ascii_lowercase:
        for b2 in string.ascii_lowercase:
            all_possible_queries.append(b1 + b2)
            
    # Add all basic numbers
    for n in range(1, 200):
        all_possible_queries.append(str(n))

    languages_queries = {}
    languages = list(base_languages_queries.keys()) 
    for lang in languages:
        languages_queries[lang] = random.sample(all_possible_queries, 40)
    
    print("Phase 1: Direct Video Search... (Gathering natively, unique video samples).", flush=True)
    
    now_utc = datetime.now(timezone.utc)
    write_header = not os.path.exists(csv_file)
    total_processed = 0
    max_videos_to_fetch = 100000
    active_csv = csv_file
    
    max_search_pages = 2 # Reduced from 10 to afford vast horizontal query expansion (far better unique channel yield per quota!)
    
    for lang, queries in languages_queries.items():
        if total_processed >= max_videos_to_fetch:
            break
            
        for query in queries:
            if total_processed >= max_videos_to_fetch:
                break
                
            q_enc = urllib.parse.quote(query)
            next_page = ""
            
            order_options = ["relevance", "date", "viewCount", "rating"]
            query_order = random.choice(order_options)

            for _ in range(max_search_pages):
                page_param = f"&pageToken={next_page}" if next_page else ""
                # Use type=video to pull specific videos immediately, randomized order
                url_template = f"https://www.googleapis.com/youtube/v3/search?key={{key}}&q={q_enc}&relevanceLanguage={lang}&type=video&part=snippet&order={query_order}&maxResults=50{page_param}"
                data = fetch_json(url_template)
                
                if data == "STOP": 
                    print("API Search Quota fully exhausted or critical error!", flush=True)
                    return
                
                batch_pairs = []
                if data and "items" in data:
                    for item in data["items"]:
                        vid = item["id"].get("videoId")
                        cid = item["snippet"].get("channelId")
                        if vid and cid and vid not in existing_video_ids:
                            existing_video_ids.add(vid)
                            batch_pairs.append((vid, cid))
                            
                # For every native batch retrieved in this Search API pull, pull the robust insights directly
                if batch_pairs:
                    vids = [p[0] for p in batch_pairs]
                    cids = [p[1] for p in batch_pairs]
                    
                    vids_enc = ",".join(vids)
                    cids_enc = ",".join(cids)
                    
                    c_url = f"https://www.googleapis.com/youtube/v3/channels?key={{key}}&id={cids_enc}&part=snippet,contentDetails,statistics"
                    c_data = fetch_json(c_url)
                    if c_data == "STOP": return
                    
                    channel_details = {}
                    if c_data and "items" in c_data:
                        for citem in c_data["items"]:
                            channel_details[citem["id"]] = citem
                            
                    v_url = f"https://www.googleapis.com/youtube/v3/videos?key={{key}}&id={vids_enc}&part=snippet,contentDetails,statistics,topicDetails"
                    v_data = fetch_json(v_url)
                    if v_data == "STOP": return
                    
                    batch_records = []
                    if v_data and "items" in v_data:
                        for vitem in v_data["items"]:
                            v_vid = vitem["id"]
                            v_snippet = vitem.get("snippet", {})
                            content_details = vitem.get("contentDetails", {})
                            stats = vitem.get("statistics", {})
                            topic_details = vitem.get("topicDetails", {})
                            
                            c_id = v_snippet.get("channelId", "")
                            c_info = channel_details.get(c_id, {})
                            c_snippet = c_info.get("snippet", {})
                            c_stats = c_info.get("statistics", {})
                            
                            published_at_str = v_snippet.get("publishedAt", "")
                            try:
                                pub_dt = datetime.strptime(published_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                upload_hour = pub_dt.hour
                                upload_day_of_week = pub_dt.weekday()
                                video_age_hours = (now_utc - pub_dt).total_seconds() / 3600.0
                            except:
                                upload_hour = None
                                upload_day_of_week = None
                                video_age_hours = None
                                
                            duration_str = content_details.get("duration", "")
                            try:
                                duration_sec = isodate.parse_duration(duration_str).total_seconds()
                            except:
                                duration_sec = None
                                
                            view_count = int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0
                            like_count = int(stats.get("likeCount", 0)) if stats.get("likeCount") else 0
                            comment_count = int(stats.get("commentCount", 0)) if stats.get("commentCount") else 0
                            
                            if view_count > 0:
                                like_view_ratio = like_count / view_count
                                comment_view_ratio = comment_count / view_count
                                engagement_score = (like_count + comment_count) / view_count
                            else:
                                like_view_ratio = 0
                                comment_view_ratio = 0
                                engagement_score = 0
                                
                            channel_published_at = c_snippet.get("publishedAt", "")
                            try:
                                c_pub_dt = datetime.strptime(channel_published_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                channel_age_days = (now_utc - c_pub_dt).total_seconds() / 86400.0
                            except:
                                channel_age_days = None

                            tags = v_snippet.get("tags", [])
                            topic_ids = topic_details.get("topicIds", [])
                            relevant_topic_ids = topic_details.get("relevantTopicIds", [])
                            
                            record = {
                                "video_id": v_vid,
                                "channel_id": c_id,
                                "channel_title": v_snippet.get("channelTitle", ""),
                                "title": v_snippet.get("title", ""),
                                "description": v_snippet.get("description", ""),
                                "tags": ",".join(tags) if tags else "",
                                "default_language": v_snippet.get("defaultLanguage", ""),
                                "default_audio_language": v_snippet.get("defaultAudioLanguage", ""),
                                "category_id": v_snippet.get("categoryId", ""),
                                "topic_ids": ",".join(topic_ids) if topic_ids else "",
                                "relevant_topic_ids": ",".join(relevant_topic_ids) if relevant_topic_ids else "",
                                "published_at": published_at_str,
                                "upload_hour": upload_hour,
                                "upload_day_of_week": upload_day_of_week,
                                "duration": duration_sec,
                                "live_broadcast_content": v_snippet.get("liveBroadcastContent", ""),
                                "view_count": view_count,
                                "like_count": like_count,
                                "comment_count": comment_count,
                                "like_view_ratio": like_view_ratio,
                                "comment_view_ratio": comment_view_ratio,
                                "engagement_score": engagement_score,
                                "subscriber_count": c_stats.get("subscriberCount", ""),
                                "channel_video_count": c_stats.get("videoCount", ""),
                                "channel_age_days": channel_age_days,
                                "channel_country": c_snippet.get("country", ""),
                                "channel_published_at": channel_published_at,
                                "positive_ratio": None,
                                "negative_ratio": None,
                                "toxic_ratio": None,
                                "related_video_ids": None,
                                "num_related_videos": 0,
                                "caption": content_details.get("caption", ""),
                                "licensed_content": content_details.get("licensedContent", False),
                                "projection": content_details.get("projection", ""),
                                "thumbnail_high_url": v_snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                                "video_age_hours": video_age_hours,
                                "collected_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                            }
                            batch_records.append(record)
                            
                    if batch_records:
                        df = pd.DataFrame(batch_records)
                        active_csv = append_to_csv_safe(df, active_csv, write_header)
                        write_header = False
                        total_processed += len(batch_records)
                        print(f"Appended +{len(batch_records)} explicitly unique direct videos! (Total: {total_processed})", flush=True)

                next_page = data.get("nextPageToken") if data else None
                if not next_page:
                    break
                time.sleep(0.01)
                
    print(f"\n--- Scraping Finished! Acquired {total_processed} new unique video samples ---\n", flush=True)

if __name__ == "__main__":
    collect_data()
