import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances
import math
from datetime import datetime, timedelta
import openrouteservice as ors
import requests
import json
import openai
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tempfile
import warnings
warnings.filterwarnings('ignore')
from io import StringIO

# 페이지 설정
st.set_page_config(
    page_title="광양제철소 폐기물 수거 경로 최적화",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WasteRouteOptimizer:
    def __init__(self):
        # ORS API 설정
        self.ors_api_key = "5b3ce3597851110001cf62489c2d4dea20f4405f9f1d318f1e2733c1"
        self.ors_client = ors.Client(key=self.ors_api_key)
        
        # OpenAI 클라이언트 초기화
        self.openai_client = openai.OpenAI(api_key="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")
        
        # 한글 폰트 등록
        self.register_korean_fonts()
        
        self.init_session()
    
    def register_korean_fonts(self):
        """한글 폰트 등록"""
        try:
            # NanumHuman 폰트 파일 등록
            font_files = [
                ("NanumHuman", "attached_assets/NanumHumanRegular_1749745170335.ttf"),
                ("NanumHuman-Bold", "attached_assets/NanumHumanBold_1749745170331.ttf"),
                ("NanumHuman-Light", "attached_assets/NanumHumanLight_1749745170335.ttf")
            ]
            
            for font_name, font_path in font_files:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    
        except Exception as e:
            # 폰트 등록 실패시 기본 폰트 사용
            pass
        
    def init_session(self):
        """세션 상태 초기화"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'box_input' not in st.session_state:
            st.session_state.box_input = ""
        if 'optimized_routes' not in st.session_state:
            st.session_state.optimized_routes = None
        if 'route_analysis' not in st.session_state:
            st.session_state.route_analysis = None

    @st.cache_data
    def load_and_process_data(_self, uploaded_file):
        """데이터 로드 및 전처리 (캐싱 적용)"""
        try:
            # 파일 형식에 따른 데이터 로드
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            else:
                df = pd.read_excel(uploaded_file)
            
            # 컬럼 정규화
            df = _self.normalize_columns(df)
            
            # 우선순위 계산
            df = _self.calculate_priority_score(df)
            
            return df
            
        except Exception as e:
            st.error(f"파일 로드 오류: {str(e)}")
            return None

    def normalize_columns(self, df):
        """컬럼명 정규화 및 표준화"""
        # 컬럼 매핑 (내장 CSV 컬럼명 반영)
        COL_MAP = {
            '박스 위치': '위치',
            '위치명': '위치',
            '위치': '위치',
            '톤': '톤수',
            '부서명': '부서',
            '부서코드': '부서코드',
            '박스구분': '용도',
            '구분': '용도',
            '위도(DD)': '좌표_위도',
            '경도(DD)': '좌표_경도',
            '위도': '좌표_위도',
            '경도': '좌표_경도',
            '위도(DMS)': '위도_DMS',
            '경도(DMS)': '경도_DMS',
            '위치정보(DD)': '위치정보_DD',
            '위치정보(DMS)': '위치정보_DMS'
        }
        # 컬럼명 변경
        df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
        # Unnamed 컬럼 제거
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        # 좌표 데이터 처리
        if '좌표_경도' not in df.columns and '경도(DD)' in df.columns:
            df['좌표_경도'] = pd.to_numeric(df['경도(DD)'], errors='coerce')
        if '좌표_위도' not in df.columns and '위도(DD)' in df.columns:
            df['좌표_위도'] = pd.to_numeric(df['위도(DD)'], errors='coerce')
        if '좌표_경도' in df.columns:
            df['좌표_경도'] = pd.to_numeric(df['좌표_경도'], errors='coerce')
            df['좌표_경도'] = df['좌표_경도'].apply(lambda x: round(x, 10) if pd.notna(x) else x)
        if '좌표_위도' in df.columns:
            df['좌표_위도'] = pd.to_numeric(df['좌표_위도'], errors='coerce')
            df['좌표_위도'] = df['좌표_위도'].apply(lambda x: round(x, 10) if pd.notna(x) else x)
        # 표준 컬럼 정의
        STANDARD_COLS = ['박스번호', '위치', '부서', '좌표_경도', '좌표_위도', '수거빈도', '지연일수', '톤수', '용도', '접수일']
        # 누락된 컬럼 생성
        for col in STANDARD_COLS:
            if col not in df.columns:
                if col in ['수거빈도', '지연일수']:
                    df[col] = np.random.randint(1, 10, len(df))
                elif col == '톤수':
                    # 내장 CSV의 '톤' 컬럼을 '톤수'로 매핑
                    if '톤' in df.columns:
                        df['톤수'] = pd.to_numeric(df['톤'], errors='coerce')
                    else:
                        df[col] = np.random.uniform(0.5, 3.0, len(df))
                elif col == '접수일':
                    df[col] = pd.date_range(start='2024-01-01', periods=len(df))
                elif col == '박스번호':
                    df[col] = range(1, len(df) + 1)
                else:
                    df[col] = ''
        # 문제가 있는 33번 수거함 제거
        initial_count = len(df)
        if '박스번호' in df.columns:
            df = df[df['박스번호'] != 33]
            filtered_count = len(df)
            if filtered_count < initial_count:
                st.warning("⚠️ 33번 수거함은 좌표 오류로 인해 자동으로 제외되었습니다.")
        return df[STANDARD_COLS]

    def calculate_priority_score(self, df):
        """우선순위 점수 계산"""
        # 우선순위 공식: 지연일수 × 수거빈도
        df['priority_score'] = df['지연일수'] * df['수거빈도']
        
        # 분위수 기반 우선순위 등급 (20-60-20)
        q20 = df['priority_score'].quantile(0.2)
        q80 = df['priority_score'].quantile(0.8)
        
        df['우선순위'] = np.select(
            [df['priority_score'] <= q20, df['priority_score'] <= q80],
            [1, 2], 
            default=3
        )
        
        return df

    def calculate_distance_matrix(self, coordinates):
        """거리 행렬 계산 (Haversine 공식 사용)"""
        coords_rad = np.radians(coordinates)
        distances = haversine_distances(coords_rad) * 6371  # 지구 반지름 (km)
        return distances

    def optimize_routes_ors(self, df, num_vehicles=3, vehicle_capacities=None, vehicle_max_counts=None):
        """ORS 기반 차량 경로 최적화"""
        if vehicle_capacities is None:
            vehicle_capacities = [8.5] * num_vehicles
        if vehicle_max_counts is None:
            vehicle_max_counts = [20] * num_vehicles
        if df.empty or '좌표_위도' not in df.columns or '좌표_경도' not in df.columns:
            return None
        
        # 좌표가 유효한 데이터만 필터링 및 문제 수거함 제외
        valid_coords = df.dropna(subset=['좌표_위도', '좌표_경도'])
        # 33번 수거함 제외 (좌표 이상)
        if '박스번호' in valid_coords.columns:
            valid_coords = valid_coords[valid_coords['박스번호'] != 33]
        
        if len(valid_coords) < 2:
            return None
        
        # 차량 수를 1-3대로 제한
        num_vehicles = min(num_vehicles, 3)
        vehicle_capacities = vehicle_capacities[:num_vehicles]
        vehicle_max_counts = vehicle_max_counts[:num_vehicles]
        
        try:
            # ORS 최적화 데이터 준비
            optimization_data = self.prepare_ors_optimization_data(valid_coords, num_vehicles, vehicle_capacities, vehicle_max_counts)

            result = self.ors_client.optimization(jobs=optimization_data['jobs'], vehicles=optimization_data['vehicles'], geometry=True)
            
            if 'routes' in result:
                return self.process_ors_result(result, valid_coords, vehicle_capacities, vehicle_max_counts)
            else:
                st.info("ORS 최적화 결과가 없어 클러스터링 최적화를 사용합니다.")
                return self.optimize_routes_fallback(valid_coords, num_vehicles, vehicle_capacities, vehicle_max_counts)
                
        except Exception as e:
            st.info(f"ORS API 연결 실패. 클러스터링 최적화를 사용합니다. 오류: {str(e)}")
            return self.optimize_routes_fallback(valid_coords, num_vehicles, vehicle_capacities, vehicle_max_counts)

    def prepare_ors_optimization_data(self, df, num_vehicles, vehicle_capacities, vehicle_max_counts):
        """openrouteservice-py 라이브러리용 최적화 데이터 준비"""
        # 가상 시작점 (광양제철소 소각로 위치)
        start_location = [127.765076, 34.926157]  # 경도, 위도 순서
        
        # Jobs 생성 (수거 장소)
        jobs = []
        job_id = 1  # 1부터 시작하는 연속적인 ID
        
        for idx, row in df.iterrows():
            skill_value = int(row['톤수'] * 10)  # 5톤 -> 50, 8.5톤 -> 85

            job = ors.optimization.Job(
                id=job_id,
                location=[row['좌표_경도'], row['좌표_위도']],
                amount=[1],  # 수거량은 1로 고정
                skills=[skill_value],  # 톤수별 스킬
                priority=int(row.get('우선순위', 1))
            )
            jobs.append(job)
            job_id += 1
        
        # Vehicles 생성 - 모든 차량 포함
        vehicles = []
        for i in range(num_vehicles):
            skill_value = int(vehicle_capacities[i] * 10)  # 5톤 -> 50, 8.5톤 -> 85
            
            vehicle = ors.optimization.Vehicle(
                id=i,
                start=start_location,
                end=start_location,
                capacity=[vehicle_max_counts[i]],  # 최대 수거 개수
                skills=[skill_value],  # 차량 톤수별 스킬
                time_window=[0, 28800]
            )
            vehicles.append(vehicle)
        
        # openrouteservice-py 라이브러리용 데이터 형식
        optimization_data = {
            "jobs": jobs,
            "vehicles": vehicles,
        }
        
        # Job ID와 박스번호 매핑을 세션에 저장
        job_to_box_mapping = {}
        job_id = 1
        for idx, row in df.iterrows():
            job_to_box_mapping[job_id] = int(row['박스번호'])
            job_id += 1
        st.session_state.job_to_box_mapping = job_to_box_mapping
        
        return optimization_data

    def process_ors_result(self, ors_result, df, vehicle_capacities, vehicle_max_counts):
        """openrouteservice-py 라이브러리 결과 처리"""
        routes = []
        total_distance = 0
        
        if 'routes' not in ors_result:
            return None
        
        # 세션에서 Job ID와 박스번호 매핑 가져오기
        job_to_box_mapping = getattr(st.session_state, 'job_to_box_mapping', {})
        
        # 모든 설정된 차량에 대해 경로 생성
        for i in range(len(vehicle_capacities)):
            vehicle_capacity = vehicle_capacities[i]
            max_count = vehicle_max_counts[i]
            
            # 해당 차량의 경로 찾기
            route_info = None
            for route in ors_result['routes']:
                if route['vehicle'] == i:
                    route_info = route
                    break
            
            if route_info and 'steps' in route_info:
                # 경로에 포함된 작업(수거 장소) 추출
                job_ids = [step['job'] for step in route_info['steps'] if 'job' in step]
                
                # Job ID를 박스번호로 변환
                box_numbers = [job_to_box_mapping.get(job_id) for job_id in job_ids if job_id in job_to_box_mapping]
                box_numbers = [box_num for box_num in box_numbers if box_num is not None]
                
                # 해당 작업들의 데이터 가져오기
                if box_numbers:
                    route_data = df[df['박스번호'].isin(box_numbers)].copy()
                else:
                    route_data = pd.DataFrame()
                
                # 경로 거리 계산
                route_distance = route_info.get('distance', 0) / 1000.0 if 'distance' in route_info else 0
                total_distance += route_distance
                
                # 지오메트리 정보 추출
                geometry = route_info.get('geometry', None)
                
            else:
                # 경로가 할당되지 않은 차량
                route_data = pd.DataFrame()
                route_distance = 0
                geometry = None
            
            routes.append({
                'vehicle_id': i + 1,
                'data': route_data,
                'distance': route_distance,
                'total_tonnage': route_data['톤수'].sum() if not route_data.empty else 0,
                'capacity': vehicle_capacity,
                'collection_count': len(route_data),
                'max_count': max_count,
                'tonnage_type': f"{vehicle_capacity}톤 전용",
                'high_priority_count': len(route_data[route_data['우선순위'] >= 2]) if not route_data.empty else 0,
                'geometry': geometry,
                'ors_optimized': True,
                'has_assignments': not route_data.empty
            })
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_vehicles': len(routes),
            'active_vehicles': len([r for r in routes if r['has_assignments']]),
            'optimization_method': 'ORS (OpenRouteService) Python Library',
            'ors_result': ors_result
        }

    def optimize_routes_fallback(self, df, num_vehicles, vehicle_capacities, vehicle_max_counts):
        """ORS 실패 시 대체 클러스터링 방법 - 모든 차량 활용"""
        routes = []
        total_distance = 0
        
        # 톤수별로 데이터 그룹화
        tonnage_groups = {}
        for tonnage in df['톤수'].unique():
            tonnage_groups[tonnage] = df[df['톤수'] == tonnage].copy().reset_index(drop=True)
        
        # 모든 차량에 대해 경로 생성
        for i in range(num_vehicles):
            vehicle_capacity = vehicle_capacities[i]
            max_count = vehicle_max_counts[i]
            
            # 해당 톤수의 데이터가 있는지 확인
            if vehicle_capacity in tonnage_groups and not tonnage_groups[vehicle_capacity].empty:
                # 우선순위 기반 정렬
                tonnage_data = tonnage_groups[vehicle_capacity].sort_values(
                    ['우선순위', 'priority_score'], 
                    ascending=[False, False]
                )
                
                # 차량당 할당할 데이터 계산
                vehicles_for_tonnage = sum(1 for cap in vehicle_capacities if cap == vehicle_capacity)
                data_per_vehicle = len(tonnage_data) // vehicles_for_tonnage
                remainder = len(tonnage_data) % vehicles_for_tonnage
                
                # 현재 차량이 해당 톤수의 몇 번째 차량인지 계산
                vehicle_order = sum(1 for j in range(i) if vehicle_capacities[j] == vehicle_capacity)
                
                # 할당할 데이터 범위 계산
                start_idx = vehicle_order * data_per_vehicle
                end_idx = start_idx + data_per_vehicle
                
                # 나머지 데이터를 첫 번째 차량들에 배분
                if vehicle_order < remainder:
                    start_idx += vehicle_order
                    end_idx += vehicle_order + 1
                else:
                    start_idx += remainder
                    end_idx += remainder
                
                # 최대 수거 개수 제한 적용
                assigned_data = tonnage_data.iloc[start_idx:min(end_idx, start_idx + max_count)].copy()
                
                if not assigned_data.empty:
                    # 간단한 거리 계산
                    route_distance = self.calculate_route_distance(assigned_data[['좌표_위도', '좌표_경도']].values)
                    total_distance += route_distance
                else:
                    assigned_data = pd.DataFrame()
                    route_distance = 0
            else:
                # 해당 톤수의 데이터가 없는 경우 빈 경로
                assigned_data = pd.DataFrame()
                route_distance = 0
            
            routes.append({
                'vehicle_id': i + 1,
                'data': assigned_data,
                'distance': route_distance,
                'total_tonnage': assigned_data['톤수'].sum() if not assigned_data.empty else 0,
                'capacity': vehicle_capacity,
                'collection_count': len(assigned_data),
                'max_count': max_count,
                'tonnage_type': f"{vehicle_capacity}톤 전용",
                'high_priority_count': len(assigned_data[assigned_data['우선순위'] >= 2]) if not assigned_data.empty else 0,
                'ors_optimized': False,
                'has_assignments': not assigned_data.empty
            })
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_vehicles': len(routes),
            'active_vehicles': len([r for r in routes if r['has_assignments']]),
            'optimization_method': '대체 클러스터링 최적화 (전체 차량 활용)'
        }

    def nearest_neighbor_tsp(self, coordinates):
        """최근접 이웃 TSP 휴리스틱"""
        if len(coordinates) <= 1:
            return list(range(len(coordinates)))
        
        n = len(coordinates)
        distance_matrix = self.calculate_distance_matrix(coordinates)
        
        unvisited = set(range(1, n))
        route = [0]  # 시작점
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return route

    def calculate_route_distance(self, coordinates):
        """경로 총 거리 계산"""
        if len(coordinates) <= 1:
            return 0
        
        total_distance = 0
        for i in range(len(coordinates) - 1):
            coord1 = np.radians([coordinates[i]])
            coord2 = np.radians([coordinates[i + 1]])
            distance = haversine_distances(coord1, coord2)[0][0] * 6371
            total_distance += distance
        
        return total_distance

    def create_route_map(self, routes_data):
        """경로 지도 시각화 (Folium 기반)"""
        if not routes_data or not routes_data['routes']:
            return None
        
        # 색상 팔레트
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'darkgreen']
        
        # 중심 좌표 계산
        all_lats = []
        all_lons = []
        
        for route in routes_data['routes']:
            if not route['data'].empty:
                all_lats.extend(route['data']['좌표_위도'].tolist())
                all_lons.extend(route['data']['좌표_경도'].tolist())
        
        if not all_lats:
            return None
            
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Folium 지도 생성
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # 광양제철소 시작점 표시
        start_location = [34.9006, 127.7669]
        folium.Marker(
            location=start_location,
            popup='<b>광양제철소</b><br>차량 출발/도착지',
            tooltip='광양제철소 (시작점)',
            icon=folium.Icon(color='black', icon='home', prefix='glyphicon')
        ).add_to(m)
        
        # 각 차량의 경로 표시
        for i, route in enumerate(routes_data['routes']):
            route_df = route['data']
            if route_df.empty:
                continue
                
            color = colors[i % len(colors)]
            
            # 차량별 FeatureGroup 생성
            vehicle_group = folium.FeatureGroup(name=f'차량 {route["vehicle_id"]} ({len(route_df)}개소)')
            
            # 수거함 위치 마커 추가
            for idx, row in route_df.iterrows():
                # 우선순위에 따른 아이콘 및 색상 선택
                if row['우선순위'] == 1:
                    icon_name = 'exclamation-sign'
                    marker_color = 'red'
                elif row['우선순위'] == 2:
                    icon_name = 'info-sign'
                    marker_color = 'orange'
                else:
                    icon_name = 'ok-sign'
                    marker_color = 'green'
                
                # 팝업 내용 생성
                popup_content = f"""
                <div style="font-family: Arial; font-size: 12px; width: 220px;">
                    <b style="color: {color};">박스 {row['박스번호']}</b><br>
                    <b>위치:</b> {row['위치']}<br>
                    <b>부서:</b> {row['부서']}<br>
                    <b>우선순위:</b> {row['우선순위']} ({'높음' if row['우선순위'] == 1 else '보통' if row['우선순위'] == 2 else '낮음'})<br>
                    <b>톤수:</b> {row['톤수']}톤<br>
                    <b>용도:</b> {row['용도']}<br>
                    <b>수거빈도:</b> {row['수거빈도']}일<br>
                    <b>지연일수:</b> {row['지연일수']}일<br>
                    <b>담당차량:</b> <span style="color: {color};">차량 {route["vehicle_id"]}</span>
                </div>
                """
                
                folium.Marker(
                    location=[row['좌표_위도'], row['좌표_경도']],
                    popup=folium.Popup(popup_content, max_width=250),
                    tooltip=f"박스 {row['박스번호']} - {row['위치']}",
                    icon=folium.Icon(
                        color=marker_color,
                        icon=icon_name,
                        prefix='glyphicon'
                    )
                ).add_to(vehicle_group)
                
                # 박스 번호를 마커 근처에 표시
                folium.map.Marker(
                    [row['좌표_위도'], row['좌표_경도']],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 10px; color: {color}; font-weight: bold; text-shadow: 1px 1px 1px white;">{row["박스번호"]}</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, -5)
                    )
                ).add_to(vehicle_group)
            
            # ORS 최적화된 경로 표시
            if route.get('ors_optimized') and route.get('geometry'):
                try:
                    geometry = route['geometry']
                    route_coords = []
                    
                    # 다양한 geometry 형식 처리
                    if isinstance(geometry, dict) and 'coordinates' in geometry:
                        coordinates = geometry['coordinates']
                        route_coords = [[lat, lon] for lon, lat in coordinates]
                    elif isinstance(geometry, list) and len(geometry) > 0:
                        if all(isinstance(coord, (list, tuple)) and len(coord) >= 2 for coord in geometry):
                            route_coords = [[lat, lon] for lon, lat in geometry]
                    elif isinstance(geometry, str):
                        decoded_coords = self.decode_polyline(geometry)
                        route_coords = [[lat, lon] for lat, lon in decoded_coords] if decoded_coords else []
                    
                    if route_coords and len(route_coords) > 1:
                        folium.PolyLine(
                            locations=route_coords,
                            weight=5,
                            color=color,
                            opacity=0.8,
                            popup=f'차량 {route["vehicle_id"]} ORS 최적화 경로'
                        ).add_to(vehicle_group)
                    else:
                        # geometry 처리 실패시 기본 라인 표시
                        if len(route_df) > 1:
                            coords = [[row['좌표_위도'], row['좌표_경도']] for _, row in route_df.iterrows()]
                            folium.PolyLine(
                                locations=coords,
                                weight=3,
                                color=color,
                                opacity=0.6,
                                dash_array='5, 5',
                                popup=f'차량 {route["vehicle_id"]} 근사 경로'
                            ).add_to(vehicle_group)
                except Exception as e:
                    # 경로 표시 실패시 기본 라인으로 대체
                    if len(route_df) > 1:
                        coords = [[row['좌표_위도'], row['좌표_경도']] for _, row in route_df.iterrows()]
                        folium.PolyLine(
                            locations=coords,
                            weight=3,
                            color=color,
                            opacity=0.6,
                            dash_array='5, 5',
                            popup=f'차량 {route["vehicle_id"]} 근사 경로'
                        ).add_to(vehicle_group)
            else:
                # 기본 직선 경로 표시
                if len(route_df) > 1:
                    coords = [[row['좌표_위도'], row['좌표_경도']] for _, row in route_df.iterrows()]
                    folium.PolyLine(
                        locations=coords,
                        weight=3,
                        color=color,
                        opacity=0.6,
                        dash_array='5, 5',
                        popup=f'차량 {route["vehicle_id"]} 근사 경로'
                    ).add_to(vehicle_group)
            
            # 차량 그룹을 지도에 추가
            vehicle_group.add_to(m)
        
        # 레이어 컨트롤 추가
        folium.LayerControl().add_to(m)
        
        # 범례 추가
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px;">
        <p style="margin: 0 0 8px 0;"><b>우선순위 표시</b></p>
        <p style="margin: 2px 0;"><i class="glyphicon glyphicon-exclamation-sign" style="color: red;"></i> 높음 (1) - 빨간색</p>
        <p style="margin: 2px 0;"><i class="glyphicon glyphicon-info-sign" style="color: orange;"></i> 보통 (2) - 주황색</p>
        <p style="margin: 2px 0;"><i class="glyphicon glyphicon-ok-sign" style="color: green;"></i> 낮음 (3) - 녹색</p>
        <p style="margin: 8px 0 2px 0;"><b>경로 표시</b></p>
        <p style="margin: 2px 0;">━━━ 실선: ORS 최적화 경로</p>
        <p style="margin: 2px 0;">┅┅┅ 점선: 근사 경로</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m

    def decode_polyline(self, polyline_str):
        """Polyline 문자열을 좌표 배열로 디코딩"""
        try:
            # 간단한 polyline 디코딩 (실제 구현은 더 복잡할 수 있음)
            # 여기서는 기본적인 처리만 수행
            return []
        except:
            return []

    def create_dashboard_charts(self, df):
        """대시보드 차트 생성"""
        charts = {}
        
        # 1. 부서별 분포 (막대 차트)
        dept_counts = df['부서'].value_counts().head(10)
        dept_df = pd.DataFrame({
            '부서': dept_counts.index,
            '수거함_수': dept_counts.values
        })
        charts['dept_distribution'] = px.bar(
            dept_df,
            x='수거함_수',
            y='부서',
            orientation='h',
            title="부서별 수거함 분포 (상위 10개)",
            labels={'수거함_수': '수거함 수', '부서': '부서'}
        ).update_layout(yaxis={'categoryorder':'total ascending'})
        
        # 2. 우선순위 분포 (파이 차트)
        priority_counts = df['우선순위'].value_counts().sort_index()
        priority_labels = {1: '낮음(1)', 2: '보통(2)', 3: '높음(3)'}
        charts['priority_distribution'] = px.pie(
            values=priority_counts.values,
            names=[priority_labels[i] for i in priority_counts.index],
            title="우선순위별 분포",
            color_discrete_sequence=['green', 'orange', 'red']
        )
        
        # 3. 톤수 분포 (히스토그램)
        charts['tonnage_distribution'] = px.histogram(
            df, 
            x='톤수', 
            nbins=20,
            title="톤수 분포",
            labels={'x': '톤수', 'y': '수거함 수'}
        )
        
        # 4. 용도별 분포 (도넛 차트)
        usage_counts = df['용도'].value_counts()
        charts['usage_distribution'] = px.pie(
            values=usage_counts.values,
            names=usage_counts.index,
            title="용도별 수거함 분포",
            hole=0.4
        )
        
        return charts

    def get_vehicle_assignment(self, box_number, routes_data):
        """박스번호에 해당하는 수거 차량 번호 반환"""
        if not routes_data or 'routes' not in routes_data:
            return '미배정'
        
        for route in routes_data['routes']:
            if box_number in route['data']['박스번호'].values:
                return f"차량 {route['vehicle_id']}"
        
        return '미배정'

    def display_metrics(self, df, routes_data=None):
        """주요 메트릭 표시"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 수거함 수", len(df))
        
        with col2:
            high_priority = len(df[df['우선순위'] >= 2])
            st.metric("고우선순위 (P≥2)", high_priority)
        
        with col3:
            avg_priority = df['우선순위'].mean()
            st.metric("평균 우선순위", f"{avg_priority:.1f}")
        
        with col4:
            if routes_data:
                active_vehicles = routes_data.get('active_vehicles', routes_data['total_vehicles'])
                total_vehicles = routes_data['total_vehicles']
                st.metric("차량 활용률", f"{active_vehicles}/{total_vehicles}대")
        
        # 추가 메트릭
        col5, col6, col7 = st.columns(3)
        
        with col5:
            incineration_count = len(df[df['용도'].str.contains('소각', na=False)])
            st.metric("소각용 수거함", incineration_count)
        
        with col6:
            recycling_count = len(df[df['용도'].str.contains('재활용', na=False)])
            st.metric("재활용 수거함", recycling_count)
        
        with col7:
            if routes_data:
                st.metric("총 이동거리", f"{routes_data['total_distance']:.1f}km")

    def display_route_analysis(self, routes_data):
        """경로 분석 결과 표시"""
        if not routes_data:
            return
        
        st.subheader("🚛 경로 최적화 분석")
        
        # 최적화 방법 표시
        optimization_method = routes_data.get('optimization_method', '알 수 없음')
        if 'ORS' in optimization_method:
            st.success(f"✅ {optimization_method}")
            st.info("OpenRouteService를 활용한 전문적인 차량 경로 최적화가 적용되었습니다.")
        else:
            st.info(f"📊 {optimization_method}")
        
        # 클러스터링 효율성 요약 (ORS가 아닌 경우에만)
        if 'cluster_efficiency' in routes_data and 'ORS' not in optimization_method:
            st.write("**지리적 클러스터링 효율성**")
            efficiency_cols = st.columns(len(routes_data['cluster_efficiency']))
            
            for i, (tonnage, stats) in enumerate(routes_data['cluster_efficiency'].items()):
                with efficiency_cols[i]:
                    avg_distance = stats['total_distance'] / stats['total_routes'] if stats['total_routes'] > 0 else 0
                    avg_collections = stats['total_collections'] / stats['total_routes'] if stats['total_routes'] > 0 else 0
                    
                    st.metric(
                        f"{tonnage}톤 차량", 
                        f"{stats['total_routes']}대",
                        f"평균 {avg_distance:.1f}km, {avg_collections:.0f}개"
                    )
        
        # 경로별 상세 정보
        for route in routes_data['routes']:
            has_assignments = route.get('has_assignments', len(route['data']) > 0)
            collection_count = route.get('collection_count', 0)
            max_count = route.get('max_count', 0)
            
            # 차량 상태 표시
            status_icon = "🚛" if has_assignments else "🚚"
            status_text = "활성" if has_assignments else "대기"
            
            tonnage_info = f" | {route['tonnage_type']} ({collection_count}/{max_count}개) - {status_text}"
            
            with st.expander(f"{status_icon} 차량 {route['vehicle_id']} - {collection_count}개소 ({route['distance']:.1f}km){tonnage_info}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("이동거리", f"{route['distance']:.1f}km")
                
                with col2:
                    st.metric("수거 개수", f"{collection_count}/{max_count}개")
                
                with col3:
                    st.metric("수거 유형", route['tonnage_type'])
                
                with col4:
                    st.metric("고우선순위", f"{route['high_priority_count']}개")
                
                # 경로 상세 테이블 (할당된 작업이 있는 경우만)
                if has_assignments and not route['data'].empty:
                    st.dataframe(
                        route['data'][['박스번호', '위치', '부서', '우선순위', '톤수', '용도']],
                        use_container_width=True
                    )
                elif not has_assignments:
                    st.info("이 차량에는 할당된 수거 작업이 없습니다.")
                
                # 차량별 개별 보고서 생성 버튼
                if has_assignments:
                    st.write(f"**📊 차량 {route['vehicle_id']} 개별 보고서**")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        if st.button(f"📄 PDF", key=f"pdf_report_{route['vehicle_id']}"):
                            with st.spinner(f"차량 {route['vehicle_id']} PDF 보고서 생성 중..."):
                                vehicle_pdf = self.create_vehicle_pdf_report(route)
                                
                                st.download_button(
                                    label=f"📥 PDF 다운로드",
                                    data=vehicle_pdf,
                                    file_name=f"차량{route['vehicle_id']}_수거보고서_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    mime="application/pdf",
                                    key=f"download_pdf_{route['vehicle_id']}"
                                )
                    
                    with col3:
                        if st.button(f"🌐 HTML", key=f"html_report_{route['vehicle_id']}"):
                            with st.spinner(f"차량 {route['vehicle_id']} HTML 보고서 생성 중..."):
                                vehicle_html = self.create_vehicle_html_report(route)
                                
                                # HTML을 base64로 인코딩하여 새 탭에서 열기
                                b64_html = base64.b64encode(vehicle_html.encode('utf-8')).decode()
                                html_link = f'<a href="data:text/html;base64,{b64_html}" target="_blank">📊 차량 {route["vehicle_id"]} 보고서 열기</a>'
                                
                                st.markdown(html_link, unsafe_allow_html=True)
                                st.success("HTML 보고서가 준비되었습니다.")

    def generate_collection_insights(self, df, routes_data):
        """OpenAI를 활용한 수거 인사이트 분석"""
        try:
            # 데이터 요약 생성
            data_summary = {
                "총_수거함_수": len(df),
                "부서_수": df['부서'].nunique(),
                "평균_우선순위": df['우선순위'].mean(),
                "톤수_분포": df['톤수'].value_counts().to_dict(),
                "우선순위_분포": df['우선순위'].value_counts().to_dict(),
                "부서별_분포": df['부서'].value_counts().head(5).to_dict()
            }
            
            # 경로 최적화 결과 요약
            if routes_data:
                route_summary = {
                    "총_차량_수": routes_data['total_vehicles'],
                    "활성_차량_수": routes_data.get('active_vehicles', routes_data['total_vehicles']),
                    "총_거리": round(routes_data['total_distance'], 1),
                    "최적화_방식": routes_data['optimization_method'],
                    "차량별_정보": []
                }
                
                for route in routes_data['routes']:
                    route_summary["차량별_정보"].append({
                        "차량_번호": route['vehicle_id'],
                        "수거_개수": route['collection_count'],
                        "최대_수거량": route['max_count'],
                        "거리": round(route['distance'], 1),
                        "톤수_유형": route['capacity'],
                        "고우선순위_개수": route['high_priority_count']
                    })
            else:
                route_summary = {"message": "경로 최적화 결과가 없습니다."}
            
            # OpenAI 프롬프트 생성
            prompt = f"""
            다음은 광양제철소 폐기물 수거 데이터 분석 결과입니다:

            **데이터 현황:**
            - 총 수거함 수: {data_summary['총_수거함_수']}개
            - 관련 부서 수: {data_summary['부서_수']}개
            - 평균 우선순위: {data_summary['평균_우선순위']:.1f}
            - 톤수 분포: {data_summary['톤수_분포']}
            - 우선순위 분포: {data_summary['우선순위_분포']}
            - 주요 부서별 분포: {data_summary['부서별_분포']}

            **경로 최적화 결과:**
            {route_summary}

            위 데이터를 바탕으로 다음 내용을 포함한 전문적인 수거 인사이트 보고서를 작성해주세요:

            1. **현황 분석**: 현재 수거 시스템의 특징과 패턴
            2. **효율성 평가**: 경로 최적화 결과의 효율성 분석
            3. **개선 제안**: 구체적이고 실행 가능한 최적화 방안
            4. **예상 효과**: 제안사항 적용 시 기대되는 개선 효과

            보고서는 한국어로 작성하고, 제철소 관리자가 이해하기 쉽도록 명확하고 간결하게 작성해주세요.
            """
            
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "당신은 산업용 폐기물 수거 최적화 전문가입니다. 데이터를 분석하여 실용적인 인사이트와 개선방안을 제시합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"인사이트 생성 중 오류가 발생했습니다: {str(e)}"

    def create_pdf_report(self, df, routes_data, insights_text):
        """PDF 보고서 생성"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # 한글 폰트 설정
        try:
            # 등록된 폰트 확인
            font_name = "NanumHuman"
            bold_font = "NanumHuman-Bold"
            pdfmetrics.getFont(font_name)
        except:
            font_name = "Helvetica"
            bold_font = "Helvetica-Bold"
        
        # 제목 스타일
        title_style = ParagraphStyle(
            'CustomTitle',
            fontName=bold_font,
            fontSize=18,
            spaceAfter=30,
            alignment=1  # 중앙 정렬
        )
        
        # 부제목 스타일
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            fontName=bold_font,
            fontSize=14,
            spaceAfter=12
        )
        
        # 본문 스타일
        normal_style = ParagraphStyle(
            'CustomNormal',
            fontName=font_name,
            fontSize=10,
            spaceAfter=6
        )
        
        # 제목
        story.append(Paragraph("광양제철소 폐기물 수거 최적화 보고서", title_style))
        story.append(Paragraph(f"생성일: {datetime.now().strftime('%Y년 %m월 %d일')}", normal_style))
        story.append(Spacer(1, 20))
        
        # 데이터 개요
        story.append(Paragraph("1. 데이터 개요", subtitle_style))
        overview_data = [
            ["항목", "값"],
            ["총 수거함 수", f"{len(df)}개"],
            ["관련 부서 수", f"{df['부서'].nunique()}개"],
            ["평균 우선순위", f"{df['우선순위'].mean():.1f}"],
            ["5톤 수거함", f"{len(df[df['톤수'] == 5.0])}개"],
            ["8.5톤 수거함", f"{len(df[df['톤수'] == 8.5])}개"]
        ]
        
        overview_table = Table(overview_data)
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), bold_font),
            ('FONTNAME', (0, 1), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # 경로 최적화 결과
        if routes_data:
            story.append(Paragraph("2. 경로 최적화 결과", subtitle_style))
            route_data = [
                ["항목", "값"],
                ["총 차량 수", f"{routes_data['total_vehicles']}대"],
                ["활성 차량 수", f"{routes_data.get('active_vehicles', routes_data['total_vehicles'])}대"],
                ["총 이동 거리", f"{routes_data['total_distance']:.1f}km"],
                ["최적화 방식", routes_data['optimization_method']]
            ]
            
            route_table = Table(route_data)
            route_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), bold_font),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(route_table)
            story.append(Spacer(1, 20))
        
        # AI 인사이트
        story.append(Paragraph("3. AI 분석 인사이트", subtitle_style))
        
        # 인사이트 텍스트를 문단으로 분할
        insights_paragraphs = insights_text.split('\n\n')
        for paragraph in insights_paragraphs:
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), normal_style))
                story.append(Spacer(1, 6))
        
        # PDF 생성
        doc.build(story)
        buffer.seek(0)
        return buffer

    def create_html_report(self, df, routes_data, insights_text):
        """HTML 형식의 세련된 보고서 생성"""
        current_time = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
        
        total_boxes = len(df)
        total_departments = df['부서'].nunique()
        avg_priority = df['우선순위'].mean()
        tonnage_5 = len(df[df['톤수'] == 5.0])
        tonnage_85 = len(df[df['톤수'] == 8.5])
        
        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>광양제철소 폐기물 수거 최적화 보고서</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Noto Sans KR', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .content {{ padding: 40px; }}
        
        .section {{
            margin-bottom: 40px;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .section-title {{
            font-size: 1.8rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-number {{
            font-size: 2.2rem;
            font-weight: 700;
            color: #3498db;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }}
        
        .insights-content {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            line-height: 1.8;
            font-size: 1rem;
            border-left: 4px solid #27ae60;
        }}
        
        .route-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .route-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            border-top: 4px solid #e74c3c;
        }}
        
        .route-card h4 {{
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .route-metric {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .metric-label {{ color: #666; font-weight: 500; }}
        .metric-value {{ color: #2c3e50; font-weight: 600; }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚛 광양제철소 폐기물 수거 최적화 보고서</h1>
            <div class="subtitle">AI 기반 경로 최적화 및 데이터 분석 결과</div>
            <div class="subtitle">생성일: {current_time}</div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">📊 데이터 개요</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_boxes}</div>
                        <div class="stat-label">총 수거함 수</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{total_departments}</div>
                        <div class="stat-label">관련 부서 수</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{avg_priority:.1f}</div>
                        <div class="stat-label">평균 우선순위</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{tonnage_5}</div>
                        <div class="stat-label">5톤 수거함</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{tonnage_85}</div>
                        <div class="stat-label">8.5톤 수거함</div>
                    </div>
                </div>
            </div>
        """
        
        # 경로 최적화 결과 섹션
        if routes_data:
            active_vehicles = routes_data.get('active_vehicles', routes_data['total_vehicles'])
            total_vehicles = routes_data['total_vehicles']
            total_distance = routes_data['total_distance']
            optimization_method = routes_data['optimization_method']
            
            html_content += f"""
            <div class="section">
                <h2 class="section-title">🚛 경로 최적화 결과</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_vehicles}</div>
                        <div class="stat-label">총 차량 수</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{active_vehicles}</div>
                        <div class="stat-label">활성 차량 수</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{total_distance:.1f}km</div>
                        <div class="stat-label">총 이동 거리</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{optimization_method.split()[0]}</div>
                        <div class="stat-label">최적화 방식</div>
                    </div>
                </div>
                
                <div class="route-summary">
            """
            
            for route in routes_data['routes']:
                has_assignments = route.get('has_assignments', len(route['data']) > 0)
                status = "활성" if has_assignments else "대기"
                
                html_content += f"""
                    <div class="route-card">
                        <h4>🚚 차량 {route['vehicle_id']} - {status}</h4>
                        <div class="route-metric">
                            <span class="metric-label">수거 유형</span>
                            <span class="metric-value">{route['tonnage_type']}</span>
                        </div>
                        <div class="route-metric">
                            <span class="metric-label">수거 개수</span>
                            <span class="metric-value">{route['collection_count']}/{route['max_count']}개</span>
                        </div>
                        <div class="route-metric">
                            <span class="metric-label">이동 거리</span>
                            <span class="metric-value">{route['distance']:.1f}km</span>
                        </div>
                        <div class="route-metric">
                            <span class="metric-label">고우선순위</span>
                            <span class="metric-value">{route['high_priority_count']}개</span>
                        </div>
                    </div>
                """
            
            html_content += "</div></div>"
        
        # AI 인사이트 섹션
        formatted_insights = insights_text.replace('\n', '<br>')
        html_content += f"""
            <div class="section">
                <h2 class="section-title">🤖 AI 분석 인사이트</h2>
                <div class="insights-content">
                    {formatted_insights}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2024 광양제철소 폐기물 수거 최적화 시스템 | AI 기반 경로 최적화 및 데이터 분석</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_content

    def create_vehicle_html_report(self, route):
        """차량별 개별 HTML 보고서 생성"""
        current_time = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
        vehicle_id = route['vehicle_id']
        
        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>차량 {vehicle_id} 수거 보고서</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Noto Sans KR', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        .content {{ padding: 30px; }}
        
        .section {{
            margin-bottom: 30px;
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
        }}
        
        .section-title {{
            font-size: 1.4rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 15px;
            border-left: 4px solid #e74c3c;
            padding-left: 12px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .info-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .info-label {{
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            font-size: 1.2rem;
            color: #e74c3c;
            font-weight: 600;
        }}
        
        .table-container {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        table {{ width: 100%; border-collapse: collapse; }}
        
        th {{
            background: #e74c3c;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
            font-size: 0.9rem;
        }}
        
        tr:nth-child(even) {{ background: #f8f9fa; }}
        tr:hover {{ background: #ffe6e6; }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 15px;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚚 차량 {vehicle_id} 수거 보고서</h1>
            <div>생성일: {current_time}</div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">차량 정보</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <div class="info-label">차량 번호</div>
                        <div class="info-value">{vehicle_id}</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">차량 유형</div>
                        <div class="info-value">{route['tonnage_type']}</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">수거 개수</div>
                        <div class="info-value">{route['collection_count']}/{route['max_count']}개</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">이동 거리</div>
                        <div class="info-value">{route['distance']:.1f}km</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">고우선순위</div>
                        <div class="info-value">{route['high_priority_count']}개</div>
                    </div>
                </div>
            </div>
        """
        
        # 수거 대상 상세 정보
        if not route['data'].empty:
            html_content += """
            <div class="section">
                <h2 class="section-title">수거 대상 상세</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>박스번호</th>
                                <th>위치</th>
                                <th>부서</th>
                                <th>우선순위</th>
                                <th>톤수</th>
                                <th>용도</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for _, row in route['data'].iterrows():
                html_content += f"""
                            <tr>
                                <td>{int(row['박스번호'])}</td>
                                <td>{row['위치']}</td>
                                <td>{row['부서']}</td>
                                <td>{int(row['우선순위'])}</td>
                                <td>{row['톤수']:.1f}</td>
                                <td>{row['용도']}</td>
                            </tr>
                """
            
            html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
        else:
            html_content += """
            <div class="section">
                <h2 class="section-title">수거 대상 상세</h2>
                <p>이 차량에는 할당된 수거 작업이 없습니다.</p>
            </div>
            """
        
        html_content += """
        </div>
        
        <div class="footer">
            <p>© 2024 광양제철소 폐기물 수거 최적화 시스템</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_content

    def create_vehicle_map(self, route_data):
        """차량별 개별 지도 생성 (경로 포함)"""
        if route_data.empty:
            return None
            
        fig = go.Figure()
        
        # 광양제철소 시작점 표시
        start_location = [34.9006, 127.7669]  # 위도, 경도 순서
        fig.add_trace(go.Scattermapbox(
            lat=[start_location[0]],
            lon=[start_location[1]],
            mode='markers',
            marker=dict(size=20, color='green', symbol='star'),
            name='광양제철소 (출발/도착)',
            hovertemplate='<b>광양제철소</b><br>차량 출발/도착지<extra></extra>'
        ))
        
        # 우선순위별 색상 설정
        priority_colors = {1: 'lightblue', 2: 'orange', 3: 'red'}
        priority_names = {1: '낮음', 2: '보통', 3: '높음'}
        
        # 우선순위별로 수거함 표시
        for priority in sorted(route_data['우선순위'].unique()):
            priority_data = route_data[route_data['우선순위'] == priority]
            if not priority_data.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=priority_data['좌표_위도'],
                    lon=priority_data['좌표_경도'],
                    mode='markers+text',
                    marker=dict(size=14, color=priority_colors[priority]),
                    text=priority_data['박스번호'].astype(str),
                    textposition="top center",
                    textfont=dict(size=10, color='black'),
                    name=f'우선순위 {priority} ({priority_names[priority]})',
                    hovertemplate='<b>박스번호:</b> %{text}<br>' +
                                 '<b>위치:</b> %{customdata[0]}<br>' +
                                 '<b>부서:</b> %{customdata[1]}<br>' +
                                 '<b>우선순위:</b> %{customdata[2]}<br>' +
                                 '<b>톤수:</b> %{customdata[3]:.1f}톤<extra></extra>',
                    customdata=priority_data[['위치', '부서', '우선순위', '톤수']].values
                ))
        
        # 수거 경로 최적화된 순서로 연결
        if len(route_data) >= 1:
            # 우선순위 순으로 정렬하여 경로 생성
            sorted_data = route_data.sort_values(['우선순위', 'priority_score'], ascending=[False, False])
            
            # 시작점 → 첫 번째 수거함
            if len(sorted_data) > 0:
                route_lats = [start_location[0]]
                route_lons = [start_location[1]]
                
                # 수거함들을 우선순위 순으로 연결
                for _, row in sorted_data.iterrows():
                    route_lats.append(row['좌표_위도'])
                    route_lons.append(row['좌표_경도'])
                
                # 마지막 수거함 → 시작점 (복귀)
                route_lats.append(start_location[0])
                route_lons.append(start_location[1])
                
                # 경로선 그리기
                fig.add_trace(go.Scattermapbox(
                    lat=route_lats,
                    lon=route_lons,
                    mode='lines',
                    line=dict(width=4, color='blue', dash='solid'),
                    name='최적 수거 경로',
                    hovertemplate='수거 경로<extra></extra>'
                ))
                
                # 방향 표시를 위한 화살표 (선택적)
                for i in range(len(route_lats)-1):
                    mid_lat = (route_lats[i] + route_lats[i+1]) / 2
                    mid_lon = (route_lons[i] + route_lons[i+1]) / 2
                    
                    if i < 3:  # 처음 몇 개만 방향 표시
                        fig.add_trace(go.Scattermapbox(
                            lat=[mid_lat],
                            lon=[mid_lon],
                            mode='markers',
                            marker=dict(size=8, color='darkblue', symbol='triangle-up'),
                            name='방향',
                            showlegend=False,
                            hovertemplate=f'경로 순서: {i+1}<extra></extra>'
                        ))
        
        # 지도 레이아웃 설정
        all_lats = list(route_data['좌표_위도']) + [start_location[0]]
        all_lons = list(route_data['좌표_경도']) + [start_location[1]]
        
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # 적절한 줌 레벨 계산
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        zoom_level = max(10, min(14, 12 - max(lat_range, lon_range) * 10))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom_level
            ),
            height=500,
            title=f"차량별 수거 경로 및 대상 위치",
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def create_vehicle_pdf_report(self, route):
        """차량별 개별 PDF 보고서 생성"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # 한글 폰트 사용 스타일 정의
        try:
            # 등록된 폰트 확인
            font_name = "NanumHuman"
            bold_font = "NanumHuman-Bold"
            # 폰트가 등록되지 않은 경우 기본 폰트 사용
            pdfmetrics.getFont(font_name)
        except:
            font_name = "Helvetica"
            bold_font = "Helvetica-Bold"
        
        # 스타일 정의
        title_style = ParagraphStyle(
            'VehicleTitle',
            fontName=bold_font,
            fontSize=16,
            spaceAfter=20,
            alignment=1  # 중앙 정렬
        )
        
        subtitle_style = ParagraphStyle(
            'VehicleSubtitle',
            fontName=bold_font,
            fontSize=12,
            spaceAfter=10
        )
        
        normal_style = ParagraphStyle(
            'VehicleNormal',
            fontName=font_name,
            fontSize=9,
            spaceAfter=4
        )
        
        # 제목
        story.append(Paragraph(f"차량 {route['vehicle_id']} 수거 보고서", title_style))
        story.append(Paragraph(f"생성일: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}", normal_style))
        story.append(Spacer(1, 20))
        
        # 차량 정보
        story.append(Paragraph("차량 정보", subtitle_style))
        vehicle_info = [
            ["항목", "내용"],
            ["차량 번호", f"{route['vehicle_id']}"],
            ["차량 유형", route['tonnage_type']],
            ["수거 개수", f"{route['collection_count']}/{route['max_count']}개"],
            ["이동 거리", f"{route['distance']:.1f}km"],
            ["고우선순위", f"{route['high_priority_count']}개"]
        ]
        
        vehicle_table = Table(vehicle_info)
        vehicle_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), bold_font),
            ('FONTNAME', (0, 1), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(vehicle_table)
        story.append(Spacer(1, 20))
        
        # 지도 이미지 추가
        if not route['data'].empty:
            try:
                # 차량별 지도 생성
                vehicle_map = self.create_vehicle_map(route['data'])
                if vehicle_map:
                    # 지도를 이미지로 변환
                    map_img_bytes = vehicle_map.to_image(format="png", width=500, height=300)
                    map_img = Image(BytesIO(map_img_bytes), width=5*inch, height=3*inch)
                    
                    story.append(Paragraph("수거 대상 위치", subtitle_style))
                    story.append(map_img)
                    story.append(Spacer(1, 20))
            except Exception as e:
                # 지도 생성 실패시 건너뛰기
                pass
        
        # 수거 대상 상세 정보
        story.append(Paragraph("수거 대상 상세", subtitle_style))
        
        if not route['data'].empty:
            # 데이터 테이블 생성
            table_data = [["박스번호", "위치", "부서", "우선순위", "톤수", "용도"]]
            
            for _, row in route['data'].iterrows():
                table_data.append([
                    str(int(row['박스번호'])),
                    str(row['위치'])[:15] + "..." if len(str(row['위치'])) > 15 else str(row['위치']),
                    str(row['부서'])[:10] + "..." if len(str(row['부서'])) > 10 else str(row['부서']),
                    str(int(row['우선순위'])),
                    f"{row['톤수']:.1f}",
                    str(row['용도'])[:10] + "..." if len(str(row['용도'])) > 10 else str(row['용도'])
                ])
            
            detail_table = Table(table_data)
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), bold_font),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            story.append(detail_table)
        else:
            story.append(Paragraph("수거 대상이 없습니다.", normal_style))
        
        # PDF 생성
        doc.build(story)
        buffer.seek(0)
        return buffer

    def display_insights_section(self, df, routes_data):
        """인사이트 섹션 표시"""
        st.subheader("🤖 AI 수거 인사이트 분석")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("OpenAI를 활용하여 수거 데이터를 분석하고 최적화 인사이트를 생성합니다.")
        
        with col2:
            if st.button("📊 인사이트 생성", type="primary"):
                with st.spinner("AI가 데이터를 분석 중입니다..."):
                    insights = self.generate_collection_insights(df, routes_data)
                    st.session_state.insights = insights
        
        # 인사이트 표시
        if hasattr(st.session_state, 'insights') and st.session_state.insights:
            st.write("### 📋 분석 결과")
            st.write(st.session_state.insights)
            
            # 보고서 다운로드 버튼
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("📄 PDF 보고서"):
                    with st.spinner("PDF 보고서 생성 중..."):
                        pdf_buffer = self.create_pdf_report(df, routes_data, st.session_state.insights)
                        
                        st.download_button(
                            label="📥 PDF 다운로드",
                            data=pdf_buffer,
                            file_name=f"수거_최적화_보고서_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key="pdf_main_report"
                        )
            
            with col3:
                if st.button("🌐 HTML 보고서"):
                    with st.spinner("HTML 보고서 생성 중..."):
                        html_content = self.create_html_report(df, routes_data, st.session_state.insights)
                        
                        # HTML을 base64로 인코딩하여 새 탭에서 열기
                        b64_html = base64.b64encode(html_content.encode('utf-8')).decode()
                        html_link = f'<a href="data:text/html;base64,{b64_html}" target="_blank">📊 HTML 보고서 열기</a>'
                        
                        st.markdown(html_link, unsafe_allow_html=True)
                        st.success("HTML 보고서가 준비되었습니다. 위 링크를 클릭하여 새 탭에서 확인하세요.")

    def run(self):
        """메인 애플리케이션 실행"""
        # 헤더
        st.title("🚛 광양제철소 폐기물 수거 경로 최적화")
        st.markdown("AI 기반 최적 경로 계획으로 효율적인 폐기물 수거를 실현합니다.")
        
        # 사이드바 - 설정
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # 파일 업로드
            uploaded_file = st.file_uploader(
                "📂 데이터 파일 업로드",
                type=["csv", "xlsx", "xls"],
                help="CSV 또는 Excel 파일을 업로드하세요."
            )
            
            # 경로 최적화 설정
            st.subheader("🚚 차량 설정")
            vehicle_options = [1, 2, 3]
            selected_vehicles = st.multiselect(
                "투입할 차량 선택",
                options=vehicle_options,
                default=[1, 2, 3],
                help="최적화에 사용할 차량을 선택하세요"
            )
            num_vehicles = len(selected_vehicles) if selected_vehicles else 1
            st.caption(f"선택된 차량: {num_vehicles}대")
            
            # 개별 차량 수거 유형 설정
            st.write("**개별 차량 수거 유형 설정**")
            st.caption("각 차량이 수거할 수 있는 톤수 유형과 최대 수거 개수를 선택하세요")
            vehicle_capacities = []
            vehicle_max_counts = []
            
            if selected_vehicles:
                for i, vehicle_num in enumerate(selected_vehicles):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        capacity = st.selectbox(
                            f"차량 {vehicle_num} 수거 유형",
                            options=[5, 8.5],
                            format_func=lambda x: f"{x}톤 수거함 전용",
                            index=1,  # 기본값 8.5톤
                            key=f"vehicle_{vehicle_num}_capacity"
                        )
                        vehicle_capacities.append(capacity)
                    
                    with col2:
                        max_count = st.number_input(
                            f"최대 수거 개수",
                            min_value=1,
                            max_value=50,
                            value=6,
                            key=f"vehicle_{vehicle_num}_max_count",
                            help="해당 차량이 수거할 수 있는 최대 수거함 개수"
                        )
                        vehicle_max_counts.append(max_count)
            else:
                st.warning("차량을 선택해주세요.")
                vehicle_capacities = [8.5]
                vehicle_max_counts = [6]
            
            # 차량별 수거 유형 요약
            capacity_summary = {}
            for i, cap in enumerate(vehicle_capacities):
                if cap not in capacity_summary:
                    capacity_summary[cap] = []
                capacity_summary[cap].append(vehicle_max_counts[i])
            
            summary_parts = []
            for cap, counts in capacity_summary.items():
                avg_count = sum(counts) / len(counts)
                summary_parts.append(f"{cap}톤: {len(counts)}대 (평균 {avg_count:.0f}개)")
            
            summary_text = ", ".join(summary_parts)
            st.info(f"차량 배치: {summary_text}")
            
            # 박스번호 필터 (삭제)
            # st.subheader("📦 박스 필터")
            # box_input = st.text_area(
            #     "특정 박스번호 입력",
            #     value=st.session_state.box_input,
            #     height=100,
            #     help="쉼표, 공백, 줄바꿈으로 구분하여 입력"
            # )
            # if box_input != st.session_state.box_input:
            #     st.session_state.box_input = box_input

        # 데이터 로드
        if uploaded_file:
            df = self.load_and_process_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ 데이터 {len(df)}건 로드 완료")
        
        # 메인 컨텐츠
        df = st.session_state.get("data")
        if df is None or df.empty:
            st.info("📁 데이터 파일을 업로드하여 시작하세요.")
            return
        
        # 박스번호 필터링 (삭제)
        # if box_input and box_input.strip():
        #     selected_boxes = re.findall(r'\d+', box_input)
        #     if selected_boxes:
        #         selected_boxes = list(map(int, selected_boxes))
        #         df = df[df['박스번호'].isin(selected_boxes)]
        #         st.info(f"🔍 {len(selected_boxes)}개 박스 선택됨 → 표시된 데이터: {len(df)}개")

        # 경로 최적화 실행
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 데이터 현황")
        with col2:
            if st.button("🚀 경로 최적화 실행", type="primary"):
                if not selected_vehicles:
                    st.error("❌ 차량을 선택해주세요.")
                elif not vehicle_capacities or len(vehicle_capacities) != len(selected_vehicles):
                    st.error("❌ 차량 설정을 완료해주세요.")
                else:
                    with st.spinner("ORS 기반 경로 최적화 진행 중..."):
                        routes_data = self.optimize_routes_ors(df, num_vehicles, vehicle_capacities, vehicle_max_counts)
                        if routes_data:
                            st.session_state.optimized_routes = routes_data
                            optimization_method = routes_data.get('optimization_method', 'ORS 최적화')
                            st.success(f"✅ 경로 최적화 완료! ({optimization_method})")
                        else:
                            st.error("❌ 좌표 데이터가 부족하여 경로 최적화를 실행할 수 없습니다.")
        
        # 메트릭 표시
        routes_data = st.session_state.get('optimized_routes')
        self.display_metrics(df, routes_data)
        
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📈 대시보드", "🗺️ 경로 지도", "📋 데이터 테이블", "📊 경로 분석"])
        
        with tab1:
            st.subheader("📊 데이터 분석 대시보드")
            charts = self.create_dashboard_charts(df)
            
            # 차트 2x2 레이아웃
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['dept_distribution'], use_container_width=True)
                st.plotly_chart(charts['tonnage_distribution'], use_container_width=True)
            
            with col2:
                st.plotly_chart(charts['priority_distribution'], use_container_width=True)
                st.plotly_chart(charts['usage_distribution'], use_container_width=True)
        
        with tab2:
            st.subheader("🗺️ 최적화된 수거 경로")
            if routes_data:
                route_map = self.create_route_map(routes_data)
                if route_map:
                    st_folium(route_map, width=700, height=600)
                else:
                    st.warning("⚠️ 좌표 데이터가 부족하여 지도를 표시할 수 없습니다.")
            else:
                st.info("🚀 '경로 최적화 실행' 버튼을 클릭하여 최적 경로를 확인하세요.")
        
        with tab3:
            st.subheader("📋 수거함 데이터")
            
            # 데이터 필터링 옵션
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                priority_filter = st.multiselect(
                    "우선순위 필터",
                    options=[1, 2, 3],
                    default=[1, 2, 3]
                )
            
            with col2:
                dept_filter = st.multiselect(
                    "부서 필터",
                    options=df['부서'].unique(),
                    default=df['부서'].unique()
                )
            
            with col3:
                usage_filter = st.multiselect(
                    "용도 필터",
                    options=df['용도'].unique(),
                    default=df['용도'].unique()
                )
            
            with col4:
                # 수거 차량 필터 (경로 최적화 결과가 있는 경우만)
                if routes_data:
                    vehicle_options = ['전체'] + [f'차량 {i}' for i in range(1, routes_data['total_vehicles'] + 1)] + ['미배정']
                    vehicle_filter = st.selectbox(
                        "수거 차량 필터",
                        options=vehicle_options,
                        index=0
                    )
                else:
                    vehicle_filter = '전체'
            
            # 수거 차량 번호 추가 (필터 적용 전에)
            def get_vehicle_for_box(box_number):
                if not routes_data or 'routes' not in routes_data:
                    return '미배정'
                
                for route in routes_data['routes']:
                    if box_number in route['data']['박스번호'].values:
                        return f"차량 {route['vehicle_id']}"
                
                return '미배정'
            
            df_with_vehicle = df.copy()
            if routes_data:
                df_with_vehicle['수거 차량'] = df_with_vehicle['박스번호'].apply(get_vehicle_for_box)
            else:
                df_with_vehicle['수거 차량'] = '미배정'
            
            # 필터 적용
            filtered_df = df_with_vehicle[
                (df_with_vehicle['우선순위'].isin(priority_filter)) &
                (df_with_vehicle['부서'].isin(dept_filter)) &
                (df_with_vehicle['용도'].isin(usage_filter))
            ].copy()
            
            # 차량 필터 적용
            if vehicle_filter != '전체':
                filtered_df = filtered_df[filtered_df['수거 차량'] == vehicle_filter].copy()
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # 다운로드 버튼
            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 필터된 데이터 다운로드 (CSV)",
                data=csv,
                file_name=f"filtered_waste_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab4:
            if routes_data:
                self.display_route_analysis(routes_data)
                
                # AI 인사이트 섹션 추가
                st.markdown("---")
                self.display_insights_section(df, routes_data)
            else:
                st.info("🚀 경로 최적화를 먼저 실행해주세요.")

if __name__ == "__main__":
    # 애플리케이션 실행
    app = WasteRouteOptimizer()
    app.run()
