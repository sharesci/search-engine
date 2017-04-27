import { Injectable } from '@angular/core'
import { Http, Response, URLSearchParams} from '@angular/http'
import { Observable } from 'rxjs';
import { AppConfig } from '../app.config.js';
import 'rxjs/add/operator/map';

@Injectable()
export class ArticleService {
    
    constructor(private _http: Http, private _config: AppConfig) { }
    
    private _articleGetterUrl = this._config.apiUrl + "/api/v1/article?";

    getArticle(id: string): Observable<any> {
        
        let queryString = new URLSearchParams();
        queryString.append('id', id);   
        
        return this._http.get(this._articleGetterUrl + queryString.toString())
                .map((response: Response) => response.json());
    }
}