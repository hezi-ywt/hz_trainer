#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import json
import re
from typing import Dict, List, Tuple, Any, Optional

class LuminaArrowPromptProcessor:
    """Lumina Arrow数据提示词处理器"""
    
    def __init__(self, 
                 replace_to_zn=0.2,
                 copyright_dropout=0.05,
                 year_dropout=0.01,
                 meta_dropout=0.01,
                 tag_dropout=0.2,
                 artist_dropout=0.05,
                 character_dropout=0.01,
                 log_fn=print):
        """
        初始化提示词处理器
        
        Args:
            replace_to_zn: 替换为中文的概率
            copyright_dropout: 版权标签丢弃概率
            year_dropout: 年份标签丢弃概率
            meta_dropout: 元数据标签丢弃概率
            tag_dropout: 标签丢弃概率
            artist_dropout: 艺术家标签丢弃概率
            character_dropout: 角色标签丢弃概率
            log_fn: 日志函数
        """
        self.replace_to_zn = replace_to_zn
        self.copyright_dropout = copyright_dropout
        self.year_dropout = year_dropout
        self.meta_dropout = meta_dropout
        self.tag_dropout = tag_dropout
        self.artist_dropout = artist_dropout
        self.character_dropout = character_dropout
        self.log_fn = log_fn
        
        # 特殊标签集合
        self.special_tags_set = {
            "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
            "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys", "male_focus",
            "1other", "2others", "3others", "4others", "5others", "6+others", "multiple_others",
        }
        
        # 系统提示词
        self.system_prompt = {
            "danbooru": "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags. <Prompt Start> ",
            "text": "You are an assistant designed to generate anime images based on textual prompts. <Prompt Start> ",
            "caption": "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> ",
            "structural_summary": "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on structural summary. <Prompt Start> ",
        }
        
        # 标签键列表
        self.tag_key_list = [
            'joycaption', 'regular_summary', 'regular_summary_zh', 'doubao_caption_dict',
            "danbooru_meta", "gemini_caption", "tags", "tag", "caption", 
            "doubao", "wd_tagger", "midjourney_style_summary", "structural_summary",
            "deviantart_commission_request", "creation_instructional_summary",
            "origin_gelbooru_data", "origin_e621_data", "origin_danbooru_data",
            "doubao_caption_dict", "gemini_caption_v2", "gemini_caption_v3",
            "gemini_caption_v4", "gemini_caption_v5", "gemini_caption_v6",
            "gemini_caption_v7", "gemini_caption_v8", "gemini_caption_v9",
            "gemini_caption_v10", "master_player_detailed_caption_en",
            "new_user_detailed_caption_en", "designer_caption_en",
            "Detailed_caption_zh", "Detailed_caption_jp", "Detailed_caption_kr",
            "Detailed_caption_en", "short_summary", "Medium_caption_en",
            "short_caption_en", "Tag_en", "Tag_zh", "short_Tag_en",
            "structured_summary_en", "Tag_mix_sentence_en", "regular_summary_en",
            "regular_summary_zh", "midjourney_style_summary_en",
            "midjourney_style_summary_zh", "midjourney_style_summary_jp",
            "creation_instructional_summary_en", "deviantart_commission_request_en",
            "main_caption_jp", "main_caption_zh", "Detailed", "Medium", "Short", "Tag",
            "Detailed_tag_en", "Detailed_tag_M_en", "Medium_tag_en", "Medium_tag_M_en",
            "Medium_tag_zh", "Short_tag_en", "Short_tag_M_en", "TagWithSentence_en",
            "TagWithSentence", "short_summary", "Medium_caption_en", "short_caption_en",
            "Detailed_caption_en", "Detailed_caption_ja", "Detailed_caption_jp",
            "Detailed_caption_kr", "Detailed_caption_zh", "Japanese_caption",
            "Japanese_translation", "Medium_caption_cn", "Medium_caption_en",
            "Medium_caption_ja", "Medium_caption_zh", "Tag_cn", "Tag_en", "Tag_ja",
            "Tag_jp", "Tag_mix_sentence_cn", "Tag_mix_sentence_en", "Tag_mix_sentence_ja",
            "Tag_mix_sentence_zh", "Tag_zh", "Translation_notes", "additional_notes",
            "alternative_summary_zh", "alternative_translation", "caption_chinese",
            "chinese_caption", "chinese_translation", "compress_nl_cn", "compress_nl_en",
            "compress_nl_ja", "compress_nl_zh", "compress_tag_cn", "compress_tag_en",
            "compress_tag_ja", "compress_tag_zh", "creation_instructional_summary_en",
            "creation_instructional_summary_ja", "creation_instructional_summary_zh",
            "danbooru_tag", "description", "designer_caption_cn", "designer_caption_en",
            "designer_caption_ja", "designer_caption_zh", "designer_caption_zh_alt",
            "detailed_description", "deviantart_commission_request_en",
            "deviantart_commission_request_ja", "deviantart_commission_request_zh",
            "image_description", "image_description_zh", "image_name", "image_url",
            "ja", "ja_translation", "japanese_caption", "japanese_translation",
            "keywords", "long_Tag_ja", "long_caption_ja", "long_caption_zh",
            "machine_translation", "machine_translation_zh",
            "master_player_detailed_caption_cn", "master_player_detailed_caption_en",
            "master_player_detailed_caption_ja", "master_player_detailed_caption_zh",
            "midjourney_style_summary_cn", "midjourney_style_summary_en",
            "midjourney_style_summary_ja", "midjourney_style_summary_jp",
            "midjourney_style_summary_zh", "new_user_detailed_caption_cn",
            "new_user_detailed_caption_en", "new_user_detailed_caption_ja",
            "new_user_detailed_caption_zh", "note", "notes", "ocr_zh", "og_prompt",
            "original_caption", "original_caption_en", "original_caption_zh",
            "original_image", "original_image_url", "original_language", "predicted_by",
            "regular_summary_cn", "regular_summary_en", "regular_summary_ja",
            "regular_summary_zh", "short_Tag_Tag_en", "short_Tag_cn", "short_Tag_en",
            "short_Tag_ja", "short_Tag_jp", "short_Tag_zh", "short_caption_cn",
            "short_caption_en", "short_caption_ja", "short_caption_jp", "short_caption_zh",
            "short_summary", "short_summary_chinese", "short_summary_ja",
            "short_summary_japanese", "short_summary_jp", "short_summary_zh",
            "short_tag_en", "short_tag_jp", "short_tag_zh", "simplified_chinese",
            "source", "source_language", "structured_summary_cn", "structured_summary_en",
            "structured_summary_ja", "structured_summary_zh", "traditional_chinese",
            "translated_by", "translated_caption_zh", "translated_notes",
            "translated_word_count", "translation", "translation_model",
            "translation_note", "translation_notes", "translation_zh", "word_by_word",
            "word_by_word_translation", "word_notes", "Tag_en", "Tag_zh",
            "short_Tag_en", "structured_summary_en", "Tag_mix_sentence_en",
            "regular_summary_en", "regular_summary_zh",
        ]

    def formate_tag(self, tag_list: List[str], rate: float = 0.8) -> List[str]:
        """格式化标签列表"""
        tag_new = []
        for tag in tag_list:
            tag = tag.strip()
            if (len(tag) > 3) and (random.random() < rate):
                tag = tag.replace("_", " ")
            tag_new.append(tag)
        return tag_new
    
    def list2str(self, tag_list: List[str]) -> str:
        """将标签列表转换为字符串"""
        tag_str = ", ".join(tag_list)
        return tag_str
    
    def str2list(self, tags_str: str) -> List[str]:
        """将标签字符串转换为列表"""
        if isinstance(tags_str, str):
            return tags_str.split(",")
        else:
            return tags_str

    def drop_tag(self, tag_list: List[str], drop_rate: float = 0.1) -> List[str]:
        """随机丢弃标签"""
        if isinstance(tag_list, list):
            new_tag_list = []
            for i in range(len(tag_list)):
                if random.random() > drop_rate:  # 保留概率为 1 - drop_rate
                    new_tag_list.append(tag_list[i])
            return new_tag_list
        else:
            return tag_list

    def extract_special_tags(self, tag_list: List[str]) -> Tuple[List[str], List[str]]:
        """提取特殊标签和一般标签"""
        special = []
        general = []
        if tag_list is None:
            return special, general
        for tag in tag_list:
            if tag in self.special_tags_set:
                special.append(tag)
            else:
                general.append(tag)
        return special, general

    def get_rating_tags(self, rating: str) -> List[str]:
        """获取评分标签"""
        try:
            if rating is None:
                return []

            rating_dict = {
                "g": ["general", "safe"], 
                "s": ["sensitive"],
                "q": ["questionable", "nsfw"], 
                "e": ["explicit", "nsfw"]
            }
            rating_list = []
            for r in rating:
                ra = rating_dict.get(r, None)
                if ra is not None:
                    rating_list.append(random.choice(ra))
            return rating_list
        except:
            return []

    def get_year_tags(self, created_at: str) -> List[str]:
        """根据创建时间获取年份标签"""
        year_tag_list = []
        try:
            if not isinstance(created_at, str) or len(created_at) < 4:
                return year_tag_list
            year_str = created_at[:4]
            if not year_str.isdigit():
                return year_tag_list
            year = int(year_str)
            if 2005 <= year <= 2010:
                year_tag = "old"
            elif 2011 <= year <= 2014:
                year_tag = "early"
            elif 2015 <= year <= 2017:
                year_tag = "mid"
            elif 2018 <= year <= 2020:
                year_tag = "recent"
            elif 2021 <= year <= 2025:
                year_tag = "newest"
            else:
                return year_tag_list
            year_tag_list.append(year_tag)
            return year_tag_list
        except:
            return year_tag_list

    def danbooru_meta_to_text(self, danbooru_meta: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str], List[str], List[str]]:
        """处理danbooru元数据"""
        character_list = danbooru_meta.get("character", [])
        artist_list = self.drop_tag(danbooru_meta.get("artist", []), 0.1)
        series_list = danbooru_meta.get("series", [])
        meta_list = danbooru_meta.get("meta", [])
        general_tag_list = self.drop_tag(danbooru_meta.get("general", []), 0.2)
        keep_tag_list = danbooru_meta.get("keep_tags", [])
        if len(keep_tag_list) > 6:
            if random.random() < 0.3:
                general_tag_list = keep_tag_list
        rating_list = danbooru_meta.get("rating_tags", [])
        quality_list = danbooru_meta.get("quality_tags", [])
        special_tag_list = danbooru_meta.get("special_tags", [])
        
        all_tag_list = list(set(special_tag_list)) + list(set(character_list)) + list(set(series_list)) + list(set(artist_list)) + list(set(general_tag_list)) + list(set(meta_list)) + list(set(rating_list)) + list(set(quality_list))
        all_tag_list = self.formate_tag(all_tag_list)
        all_tag_list = self.drop_tag(all_tag_list)
        all_tag_text = ", ".join(all_tag_list)
        return all_tag_text, character_list, artist_list, series_list, rating_list, quality_list

    def origin_danbooru_data_to_text(self, origin_danbooru_data: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str], List[str], List[str]]:
        """处理原始danbooru数据"""
        artist_list = origin_danbooru_data.get("artist_tags", [])
        artist_list = self.drop_tag(artist_list, self.artist_dropout)
        character_list = origin_danbooru_data.get("character_tags", [])
        character_list = self.drop_tag(character_list, self.character_dropout)
        copyright_list = origin_danbooru_data.get("copyright_tags", [])
        copyright_list = self.drop_tag(copyright_list, self.copyright_dropout)
        origin_general_tag_list = origin_danbooru_data.get("general_tags", [])
        origin_general_tag_list = self.drop_tag(origin_general_tag_list, self.tag_dropout)
        special_tag_list, general_tag_list = self.extract_special_tags(origin_general_tag_list)
 
        rating_list = self.get_rating_tags(origin_danbooru_data.get("rating", None))
        rating_list = self.drop_tag(rating_list, self.tag_dropout)
        year_list = self.get_year_tags(origin_danbooru_data.get("created_at", None))
        year_list = self.drop_tag(year_list, self.year_dropout)

        meta_list = origin_danbooru_data.get("meta_tags", [])
        lists_to_combine = [
            special_tag_list, character_list, copyright_list, artist_list,
            general_tag_list, rating_list, meta_list, year_list
        ]

        all_tag_list = []
        for lst in lists_to_combine:
            if lst is not None:
                all_tag_list.extend(list(set(lst)))
        all_tag_list = [x for x in all_tag_list if len(x) > 0]
        all_tag_list = self.formate_tag(all_tag_list)
        if random.random() < 0.001:
            all_tag_list = list(set(all_tag_list))
        all_tag_text = ", ".join(all_tag_list)
        return all_tag_text, character_list, artist_list, copyright_list, rating_list, []

    def origin_gelbooru_data_to_text(self, origin_gelbooru_data: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str], List[str], List[str]]:
        """处理原始gelbooru数据"""
        artist_list = origin_gelbooru_data.get("artist_tags", [])
        artist_list = self.drop_tag(artist_list, self.artist_dropout)
        character_list = origin_gelbooru_data.get("character_tags", [])
        character_list = self.drop_tag(character_list, self.character_dropout)
        copyright_list = origin_gelbooru_data.get("copyright_tags", [])
        copyright_list = self.drop_tag(copyright_list, self.copyright_dropout)
        origin_general_tag_list = origin_gelbooru_data.get("general_tags", [])
        origin_general_tag_list = self.drop_tag(origin_general_tag_list, self.tag_dropout)
        special_tag_list, general_tag_list = self.extract_special_tags(origin_general_tag_list)
        rating_list = self.get_rating_tags(origin_gelbooru_data.get("rating", None))
        rating_list = self.drop_tag(rating_list, self.tag_dropout)
        year_list = self.get_year_tags(origin_gelbooru_data.get("created_at", None))
        year_list = self.drop_tag(year_list, self.year_dropout)
        meta_list = origin_gelbooru_data.get("meta_tags", [])
        
        lists_to_combine_dict = {
            "special_tag_list": special_tag_list, "character_list": character_list,
            "copyright_list": copyright_list, "artist_list": artist_list,
            "general_tag_list": general_tag_list, "rating_list": rating_list,
            "meta_list": meta_list, "year_list": year_list
        }

        all_tag_list = []
        for key, lst in lists_to_combine_dict.items():
            if lst is not None:
                all_tag_list.extend(list(set(lst)))

        all_tag_list = [x for x in all_tag_list if len(x) > 0]
        all_tag_list = self.formate_tag(all_tag_list)
        if random.random() < 0.001:
            all_tag_list = list(set(all_tag_list))
        all_tag_text = ", ".join(all_tag_list)

        return all_tag_text, character_list, artist_list, copyright_list, rating_list, []

    def origin_e621_data_to_text(self, origin_e621_data: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str], List[str], List[str]]:
        """处理原始e621数据"""
        artist_list = origin_e621_data.get("artist_tags", [])
        artist_list = self.drop_tag(artist_list, self.artist_dropout)
        character_list = origin_e621_data.get("character_tags", [])
        character_list = self.drop_tag(character_list, self.character_dropout)
        copyright_list = origin_e621_data.get("copyright_tags", [])
        copyright_list = self.drop_tag(copyright_list, self.copyright_dropout)
        general_tag_list = origin_e621_data.get("general_tags", [])
        general_tag_list = self.drop_tag(general_tag_list, self.tag_dropout)
        species_tag_list = origin_e621_data.get("species_tags", [])
        rating_list = self.get_rating_tags(origin_e621_data.get("rating", None))
        rating_list = self.drop_tag(rating_list, self.tag_dropout)
        year_list = self.get_year_tags(origin_e621_data.get("created_at", None))
        year_list = self.drop_tag(year_list, self.year_dropout)
        special_tag_list = origin_e621_data.get("special_tags", [])
        meta_list = origin_e621_data.get("meta_tags", [])
        meta_list = self.drop_tag(meta_list, self.meta_dropout)
        
        lists_to_combine = [
            special_tag_list, character_list, copyright_list, artist_list,
            species_tag_list, general_tag_list, rating_list, meta_list, year_list
        ]

        all_tag_list = []
        for lst in lists_to_combine:
            if lst is not None:
                all_tag_list.extend(list(set(lst)))
 
        all_tag_list = [x for x in all_tag_list if len(x) > 0]
        all_tag_list = self.formate_tag(all_tag_list)
        if random.random() < 0.001:
            all_tag_list = list(set(all_tag_list))
        all_tag_text = ", ".join(all_tag_list)
        return all_tag_text, character_list, artist_list, copyright_list, rating_list, []

    def build_system_prompt(self, type: str, artist_list: List[str], series_list: List[str], rating_list: List[str], quality_list: List[str]) -> str:
        """构建系统提示词"""
        if len(artist_list) > 0:
            if type == "danbooru_meta":
                system_prompt = f"You are an artist named @{', '.join(artist_list)}, you need to create works in your own style with the highest degree of image-text alignment based on danbooru tags, the danbooru tag may include the character, the artist style, the action, etc. <Prompt Start>  "
            elif type == "text":
                system_prompt = f"You are an artist named @{', '.join(artist_list)}, you need to create works in your own style based on textual prompts. <Prompt Start>  "
        else:
            system_prompt = self.system_prompt[type]
            
        return system_prompt

    def add_character_artist(self, character_list: List[str], artist_list: List[str], user_prompt: str) -> str:
        """添加角色和艺术家信息到提示词"""
        if isinstance(character_list, list) and len(character_list) > 0:
            character_list = [f"#{character}" for character in character_list if character != "" and character != " "]
            random.shuffle(character_list)
        else:
            character_list = None
            
        if isinstance(artist_list, list) and len(artist_list) > 0:
            artist_list = [f"@{artist}" for artist in artist_list if artist != ""]
            random.shuffle(artist_list)
        else:
            artist_list = None
            
        add = ""

        if character_list is not None:
            if len(character_list) > 0:
                character_list = ", ".join(character_list)
                type_list = [
                    f"Characters: {character_list}.",
                    f"Cast: {character_list}.",
                    f"The characters in this work including {character_list}.",
                    f"{character_list}",
                ]
                add = add + random.choice(type_list)
            
        if artist_list is not None:
            if len(artist_list) > 0:
                artist_list = ",".join(artist_list)
                type_list = [
                    f"Drawn by {artist_list}.",
                    f"Painted by {artist_list}.",
                    f"Created by {artist_list}.",
                    f"Artist: {artist_list}.",
                    f"A vision of {artist_list}.",
                    f"This work is attributed to {artist_list}.",
                    f"Use {artist_list} style.",
                    f"{artist_list}",
                ]
                if add != "":
                    if random.random() < 0.5:
                        if random.random() < 0.5:
                            add = "\n" + add + "\n" + random.choice(type_list)
                        else:
                            add = "\n" + random.choice(type_list) + "\n" + add
                    else:
                        if random.random() < 0.5:
                            add = add + " " + random.choice(type_list)
                        else:
                            add = random.choice(type_list) + " " + add
                else:
                    add = add + random.choice(type_list)
                        
        if add != "":
            if random.random() < 0.5:
                if random.random() < 0.5:
                    user_prompt = add + "\n" + user_prompt
                else:
                    user_prompt = user_prompt + " " + add
            else:
                if random.random() < 0.5:
                    user_prompt = user_prompt + " " + add
                else:
                    user_prompt = add + " " + user_prompt
        return user_prompt

    def process_text_zh(self, json_data: Dict[str, Any]) -> str:
        """处理text_zh字段，提取并生成最终的提示词"""
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
            
        caption_dict = {}
        meta_has = False
        character_list = []
        artist_list = []
        series_list = []
        rating_list = []
        quality_list = []
        
        for tag_key in self.tag_key_list:
            if tag_key in json_data and json_data[tag_key] is not None:
                if len(json_data[tag_key]) > 0:
                    if tag_key == 'danbooru_meta':
                        all_tag_text, character_list, artist_list, series_list, rating_list, quality_list = self.danbooru_meta_to_text(json_data[tag_key])
                        meta_has = True
                        if isinstance(all_tag_text, str):
                            if random.random() < 0.2:
                                caption_dict[tag_key] = [self.system_prompt["danbooru"], all_tag_text, tag_key]
                            else:
                                caption_dict[tag_key] = [self.system_prompt["text"], all_tag_text, tag_key]
                        else:
                            continue
                            
                    elif tag_key == 'origin_danbooru_data':
                        all_tag_text, character_list, artist_list, series_list, rating_list, quality_list = self.origin_danbooru_data_to_text(json_data[tag_key])
                        meta_has = True
                        if isinstance(all_tag_text, str) and len(all_tag_text) > 10:
                            if random.random() < 0.01:
                                caption_dict[tag_key] = [self.system_prompt["danbooru"], all_tag_text, tag_key]
                            else:
                                caption_dict[tag_key] = [self.system_prompt["text"], all_tag_text, tag_key]
                                
                    elif tag_key == 'origin_gelbooru_data':
                        all_tag_text, character_list, artist_list, series_list, rating_list, quality_list = self.origin_gelbooru_data_to_text(json_data[tag_key])
                        meta_has = True
                        if isinstance(all_tag_text, str) and len(all_tag_text) > 10:
                            if random.random() < 0.01:
                                caption_dict[tag_key] = [self.system_prompt["danbooru"], all_tag_text, tag_key]
                            else:
                                caption_dict[tag_key] = [self.system_prompt["text"], all_tag_text, tag_key]
                                caption_dict[tag_key+"_v2"] = [self.system_prompt["text"], all_tag_text, tag_key]
                                
                    elif tag_key == 'origin_e621_data':
                        all_tag_text, character_list, artist_list, series_list, rating_list, quality_list = self.origin_e621_data_to_text(json_data[tag_key])
                        meta_has = True
                        if isinstance(all_tag_text, str) and len(all_tag_text) > 10:
                            if random.random() < 0.01:
                                caption_dict[tag_key] = [self.system_prompt["danbooru"], all_tag_text, tag_key]
                            else:
                                caption_dict[tag_key] = [self.system_prompt["text"], all_tag_text, tag_key]
                                
                    elif tag_key == 'wd_tagger':
                        if len(json_data[tag_key]) > 4:
                            tags = json_data[tag_key].replace("|||", "")
                            tags = self.str2list(tags)
                            tags = self.formate_tag(tags)
                            tags = self.list2str(tags)
                            if random.random() < 0.01:
                                caption_dict[tag_key] = [self.system_prompt["danbooru"], tags, tag_key]
                            else:
                                caption_dict[tag_key] = [self.system_prompt["text"], tags, tag_key]
                                
                    elif 'gemini_caption' in tag_key:
                        gemini_caption = json_data[tag_key]
                        
                        if isinstance(gemini_caption, dict):
                            for sub_tag_key in self.tag_key_list:
                                if sub_tag_key in gemini_caption and gemini_caption[sub_tag_key] is not None:
                                    if isinstance(gemini_caption[sub_tag_key], str):
                                        if len(gemini_caption[sub_tag_key]) > 15:
                                            key = tag_key + "_" + sub_tag_key
                                            if sub_tag_key == "regular_summary" or sub_tag_key == "Detailed":
                                                if random.random() < 0.1:
                                                    caption_dict[key] = [self.system_prompt["caption"], gemini_caption[sub_tag_key], key]
                                                else:
                                                    caption_dict[key] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]
                                                
                                                if random.random() < 0.1:
                                                    caption_dict[key+"_v2"] = [self.system_prompt["caption"], gemini_caption[sub_tag_key], key]
                                                else:
                                                    caption_dict[key+"_v2"] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]
                                                if random.random() < 0.1:
                                                    caption_dict[key+"_v3"] = [self.system_prompt["caption"], gemini_caption[sub_tag_key], key]
                                                else:
                                                    caption_dict[key+"_v3"] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]
                                            else:
                                                caption_dict[key] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]
                        
                        elif isinstance(gemini_caption, str):
                            if len(gemini_caption) > 30:
                                caption_dict[tag_key] = [self.system_prompt["text"], gemini_caption, tag_key]
                        else:
                            continue
                            
                    elif tag_key == "gemini_caption":
                        gemini_caption = json_data[tag_key]
                        
                        if isinstance(gemini_caption, dict):
                            for sub_tag_key in self.tag_key_list:
                                if sub_tag_key in gemini_caption and gemini_caption[sub_tag_key] is not None:
                                    if len(gemini_caption[sub_tag_key]) > 30:
                                        key = tag_key + "_" + sub_tag_key
                                        if sub_tag_key == "regular_summary":
                                            if random.random() < 0.5:
                                                caption_dict[key] = [self.system_prompt["caption"], gemini_caption[sub_tag_key], key]
                                            else:
                                                caption_dict[key] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]                           
                                        else:
                                            caption_dict[key] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]
                                            
                    elif tag_key == 'doubao_caption_dict':
                        gemini_caption = json_data[tag_key]
                        if isinstance(gemini_caption, dict):
                            for sub_tag_key in self.tag_key_list:
                                if sub_tag_key in gemini_caption and gemini_caption[sub_tag_key] is not None:
                                    if len(gemini_caption[sub_tag_key]) > 30:
                                        key = tag_key + "_" + sub_tag_key
                                        if sub_tag_key == "regular_summary":
                                            if random.random() < 0.05:
                                                caption_dict[key] = [self.system_prompt["caption"], gemini_caption[sub_tag_key], key]
                                            else:
                                                caption_dict[key] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]
                                        else:
                                            caption_dict[key] = [self.system_prompt["text"], gemini_caption[sub_tag_key], key]
                        elif isinstance(gemini_caption, str):
                            if len(gemini_caption) > 30:
                                caption_dict[tag_key] = [self.system_prompt["text"], gemini_caption, tag_key]
                        else:
                            continue
                            
                    elif tag_key == 'structural_summary':
                        if isinstance(json_data[tag_key], str):
                            if random.random() < 0.01:
                                caption_dict[tag_key] = [self.system_prompt["structural_summary"], json_data[tag_key], tag_key]
                            else:
                                caption_dict[tag_key] = [self.system_prompt["text"], json_data[tag_key], tag_key]
                        else:
                            continue
                    else:
                        if len(json_data[tag_key]) > 20 and isinstance(json_data[tag_key], str):
                            caption_dict[tag_key] = [self.system_prompt["text"], json_data[tag_key], tag_key]
                        else:
                            continue
                        
        if len(caption_dict) == 0:
            self.log_fn(f"process_text_zh | No caption found, use default caption")
            if random.random() < 0.5:
                caption_dict["default"] = ['You are an assistant designed to generate high quality anime images based on textual prompts. <Prompt Start>  ', 'Generate a random anime image', 'default']
            else:
                caption_dict["default"] = ['', '', 'default']

        text = random.choice(list(caption_dict.values()))

        if random.random() < 0.001:
            if meta_has:
                try:
                    caption = self.build_system_prompt(text[2], artist_list, series_list, rating_list, quality_list) + text[0] + text[1]
                except:
                    caption = text[0] + text[1]
            else:
                caption = text[0] + text[1]
        else:
            if random.random() < 0.3:
                caption = text[0] + text[1]
            else:
                if meta_has and "gemini_caption_v2" not in text[2]:
                    caption = text[0] + self.add_character_artist(character_list, artist_list, text[1])
                else:
                    caption = text[0] + text[1]

        if random.random() < 0.01:
            self.log_fn(f"process_text_zh | type: {text[2]} | text: {caption}")

        return caption


def process_single_text_zh(text_zh_data: str, **kwargs) -> str:
    """处理单个text_zh数据的便捷函数"""
    processor = LuminaArrowPromptProcessor(**kwargs)
    return processor.process_text_zh(text_zh_data)


if __name__ == "__main__":
    # 测试示例
    import json
    
    # 示例text_zh数据
    sample_text_zh = {
        "gemini_caption_v6": {
            "midjourney_style_summary_zh": "魅惑眼神，柔媚肌肤，Raiden Shogun躺卧姿态，白色绷带缠绕，花瓣点缀，性感氛围"
        },
        "wd_tagger": "1girl, long_hair, blue_eyes, white_hair, dress, sitting, flower",
        "wd_tagger_metadata": {"confidence": 0.95}
    }
    
    # 创建处理器
    processor = LuminaArrowPromptProcessor()
    
    json_path = '/mnt/huggingface/add_arrow4/all_text_zh1.json'
    prompt_list = []
    with open(json_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        result = processor.process_text_zh(item)
        prompt_list.append(result)

    # #split to 4 files
    # for i in range(4):
    #     with open(f'/mnt/huggingface/add_arrow4/all_text_zh_prompt_{i}.json', 'w', encoding='utf-8') as f:
    #         json.dump(prompt_list[i::4], f, ensure_ascii=False, indent=2)

    with open('/mnt/huggingface/add_arrow4/all_text_zh1_prompt.json', 'w', encoding='utf-8') as f:
        json.dump(prompt_list, f, ensure_ascii=False, indent=2)

    # # 处理数据
    # result = processor.process_text_zh(sample_text_zh)
    # print(f"处理结果: {result}")
    
    # # 测试便捷函数
    # result2 = process_single_text_zh(json.dumps(sample_text_zh))
    # print(f"便捷函数结果: {result2}")
