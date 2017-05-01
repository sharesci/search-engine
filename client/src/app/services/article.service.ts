import { Injectable } from '@angular/core'
import { Http, Response, URLSearchParams, ResponseContentType } from '@angular/http'
import { Observable } from 'rxjs';
import { AppConfig } from '../app.config.js';
import 'rxjs/add/operator/map';

@Injectable()
export class ArticleService {

    constructor(private _http: Http, private _config: AppConfig) { }

    private _articleUrl = this._config.apiUrl + "/api/v1/article?";

    getArticle(id: string, pdf: boolean): Observable<any> {
        let queryString = new URLSearchParams();
        queryString.append('id', id);
        if (pdf) {
            queryString.append('pdf', '1');
            return this._http.get(this._articleUrl + queryString.toString(), { responseType: ResponseContentType.Blob })
                .map((res) => { return new Blob([res.blob()], { type: 'application/pdf' })})
            }
        
        return this._http.get(this._articleUrl + queryString.toString())
            .map((response: Response) => response.json());
    }
}